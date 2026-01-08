import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# ==========================================
# 1. 核心参数
# ==========================================
SPHERE_CENTER = np.array([0.0, 0.35, 1.3])
SPHERE_R = 1.3
ROBOT_REACH = 0.82
EE_SITE_NAME = "ee_site"

button_left, button_right = False, False
lastx, lasty = 0, 0

def mouse_button(window, button, act, mods):
    global button_left, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

def mouse_move(window, xpos, ypos):
    global lastx, lasty
    dx, dy = xpos - lastx, ypos - lasty
    lastx, lasty = xpos, ypos
    if not (button_left or button_right): return
    action = mj.mjtMouse.mjMOUSE_ROTATE_V if button_left else mj.mjtMouse.mjMOUSE_MOVE_V
    mj.mjv_moveCamera(model, action, dx/800, dy/800, scene, cam)

def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scene, cam)

# ==========================================
# 2. 映射逻辑 (抬高目标点，避免物理限制)
# ==========================================
def project_to_reachable_sphere(nx, ny, lift_offset=0):
    target_x = nx * 0.25
    target_y = -0.1 + (ny * 0.25)
    target_z = 0.85 
    
    virtual_p = np.array([target_x, target_y, target_z])
    direction = virtual_p - SPHERE_CENTER
    direction /= np.linalg.norm(direction)
    p_final = SPHERE_CENTER + direction * (SPHERE_R - lift_offset)
    
    if p_final[2] < 0.05: p_final[2] = 0.05
    dist_to_base = np.linalg.norm(p_final)
    if dist_to_base > ROBOT_REACH:
        p_final = (p_final / dist_to_base) * ROBOT_REACH
    return p_final

# ==========================================
# 3. 强化版 IK 控制器 (强制 Elbow-Up)
# ==========================================
def IK_controller(model, data, target_pos, target_quat, site_id):
    cur_pos = data.site_xpos[site_id]
    cur_mat = data.site_xmat[site_id].reshape(3, 3)
    cur_quat = np.zeros(4)
    mj.mju_mat2Quat(cur_quat, cur_mat.flatten())
    
    pos_err = target_pos - cur_pos
    res_quat = np.zeros(4)
    mj.mju_negQuat(res_quat, cur_quat)
    rot_err_quat = np.zeros(4)
    mj.mju_mulQuat(rot_err_quat, target_quat, res_quat)
    rot_err = rot_err_quat[1:] * np.sign(rot_err_quat[0])
    
    err = np.concatenate([pos_err, rot_err])
    jac = np.zeros((6, model.nv))
    mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    # 求解基础 dq
    jac_inv = np.linalg.pinv(jac, rcond=0.01)
    dq = jac_inv @ err
    
    # --- 核心改进：零空间姿态偏置 ---
    # 强制期望 Elbow 为正 (1.57), Shoulder 稍微靠后 (-1.0)
    # 这会产生一个“向上”的拉力
    q_ref = np.array([data.qpos[0], -1.0, 1.57, -1.57, -1.57, 0])
    null_space_proj = np.eye(model.nv) - jac_inv @ jac
    dq += null_space_proj @ (0.5 * (q_ref - data.qpos[:6])) # 加大偏置系数
    
    max_dq = 0.5
    dq_norm = np.linalg.norm(dq)
    if dq_norm > max_dq: dq *= (max_dq / dq_norm)
    return dq

# ==========================================
# 4. 初始化与计划
# ==========================================
xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/universal_robots_ur5e/scene.xml'))
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
if sid == -1: sid = 0

strokes = np.load("stroke_data.npy", allow_pickle=True)
all_pts = np.vstack([s for s in strokes if len(s) > 1])
center, scale = (all_pts.min(0) + all_pts.max(0))/2, np.max(all_pts.max(0)-all_pts.min(0))

full_plan = []
for stroke in strokes:
    if len(stroke) < 2: continue
    pts = [project_to_reachable_sphere((p[0]-center[0])/(scale/2), -(p[1]-center[1])/(scale/2)) for p in stroke]
    full_plan.append((pts[0], pts[0], 0.1, False)) 
    for i in range(len(pts)-1):
        full_plan.append((pts[i], pts[i+1], 0.01, True)) 
    full_plan.append((pts[-1], pts[-1], 0.1, False))

cum_time = np.cumsum([0] + [p[2] for p in full_plan])

# 初始位姿：强制高耸
data.qpos[:] = [0, -0.8, 1.8, -2.5, -1.57, 0]
mj.mj_forward(model, data)

# ==========================================
# 5. GLFW & 主循环
# ==========================================
glfw.init()
window = glfw.create_window(1200, 900, "Forced Elbow-Up Solution", None, None)
glfw.make_context_current(window)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_scroll_callback(window, scroll)

scene = mj.MjvScene(model, maxgeom=150000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
cam = mj.MjvCamera(); mj.mjv_defaultCamera(cam)
cam.lookat = [0, 0, 0.8]; cam.distance = 2.5

traj_points = []

while not glfw.window_should_close(window):
    if data.time < cum_time[-1]:
        for _ in range(40):
            if data.time >= cum_time[-1]: break
            idx = np.searchsorted(cum_time, data.time, side='right') - 1
            p0, p1, dur, is_writing = full_plan[np.clip(idx, 0, len(full_plan)-1)]
            tau = np.clip((data.time - cum_time[idx])/dur, 0, 1)
            target_pos = p0 + (p1 - p0) * (tau**2 * (3 - 2 * tau))
            
            z_dir = target_pos - SPHERE_CENTER; z_dir /= np.linalg.norm(z_dir)
            y_tmp = np.array([0, 0, 1]); x_dir = np.cross(y_tmp, z_dir); x_dir /= np.linalg.norm(x_dir)
            y_dir = np.cross(z_dir, x_dir); target_quat = np.zeros(4)
            mj.mju_mat2Quat(target_quat, np.column_stack([x_dir, y_dir, z_dir]).flatten())
            
            old_q = data.qpos.copy()
            dq = IK_controller(model, data, target_pos, target_quat, sid)
            data.qpos[:] += dq * 0.8
            data.qvel[:] = (data.qpos - old_q) / (model.opt.timestep * 10)
            mj.mj_step(model, data)
            if is_writing: traj_points.append(data.site_xpos[sid].copy())
    else:
        mj.mj_step(model, data)

    viewport = mj.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, 3, scene)
    mj.mjv_initGeom(scene.geoms[scene.ngeom], 3, [SPHERE_R, 0, 0], SPHERE_CENTER, np.eye(3).flatten(), [1, 0, 0, 0.05])
    scene.ngeom += 1

    for p in traj_points[::20]:
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(scene.geoms[scene.ngeom], 2, [0.005, 0, 0], p, np.eye(3).flatten(), [1, 1, 1, 1])
            scene.ngeom += 1

    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()