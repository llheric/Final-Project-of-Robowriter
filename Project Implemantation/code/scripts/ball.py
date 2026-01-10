import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心参数
# ==========================================
SPHERE_CENTER = np.array([0.0, 0.35, 1.3])
SPHERE_R = 1.3  # 球面半径
ROBOT_REACH = 0.82 # 机械臂最大可达距离
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
    # --- 1. 控制字的大小 ---
    # 0.2 左右在侧壁投影会显得比较合适
    scale_factor = 0.2  
    
    # --- 2. 构造侧壁引导点 ---
    # 我们让目标点位于球体前方（Y轴增加）且稍微偏上的位置
    target_x = nx * scale_factor
    # 这里的 0.65 (球心0.35 + 0.3) 让投影点往球面的斜前方靠，从而获得弧度
    target_y = (ny * scale_factor) + 0.65 
    # 引导点设为 0.05，确保它是向斜下方投影
    target_z = 0.05 
    
    virtual_p = np.array([target_x, target_y, target_z])
    direction = virtual_p - SPHERE_CENTER
    direction /= np.linalg.norm(direction)
    
    # --- 3. 投影到球面 ---
    # lift_offset 为正时提笔（向球心收缩），为 0 时落笔（在球面上）
    p_final = SPHERE_CENTER + direction * (SPHERE_R - lift_offset)
    
    # --- 4. 严格 z 范围限制 ---
    # 如果投影点太高，我们将其压低到 0.1，但此时它仍带有球面的 XY 弧度
    if p_final[2] > 0.1: 
        p_final[2] = 0.1
    if p_final[2] < 0.01: 
        p_final[2] = 0.01
        
    # 可达性检查
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
xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'D:/MuHaoyan ZhangJiyun LiuLihe _Final Project/Project Implemantation/code/models/universal_robots_ur5e/scene.xml'))
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, EE_SITE_NAME)  # 获取末端执行器站点ID
if sid == -1: sid = 0

strokes = np.load("stroke_data.npy", allow_pickle=True)
all_pts = np.vstack([s for s in strokes if len(s) > 1])
center, scale = (all_pts.min(0) + all_pts.max(0))/2, np.max(all_pts.max(0)-all_pts.min(0))

full_plan = []
LIFT_VAL = 0.04  # 提笔高度 (5cm)
WAIT_DUR = 0.15   # 提笔/落笔动作耗时

for stroke in strokes:
    if len(stroke) < 2: continue
    
    # 1. 归一化笔画
    norm_pts = [((p[0]-center[0])/(scale/2), -(p[1]-center[1])/(scale/2)) for p in stroke]
    
    # 2. 生成对应的球面坐标 (落笔在 z 附近) 和 提笔坐标 (远离球面)
    # 注意：在底部书写时，提笔 offset 会让笔尖更靠近地面（如果 z 值减小）
    # 或者我们可以在 project 函数里逻辑反转。
    # 为了简单，我们直接生成：
    pts_on_sphere = [project_to_reachable_sphere(nx, ny, lift_offset=0) for nx, ny in norm_pts]
    pts_lifted = [project_to_reachable_sphere(nx, ny, lift_offset=LIFT_VAL) for nx, ny in norm_pts]

    # 落笔动作：从空中下降到球面起点
    full_plan.append((pts_lifted[0], pts_on_sphere[0], WAIT_DUR, False)) 
    
    # 书写动作：严格在球面上
    for i in range(len(pts_on_sphere)-1):
        full_plan.append((pts_on_sphere[i], pts_on_sphere[i+1], 0.01, True)) 
    
    # 提笔动作：从终点抬起到空中
    full_plan.append((pts_on_sphere[-1], pts_lifted[-1], WAIT_DUR, False))

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
# 新增：用于绘图的数据容器
joint_data_log = []  # 存储 qpos
time_log = []        # 存储对应的仿真时间

# --- 1. 在循环外定义固定参数 ---
Q_HOME = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0])
# --- 建议在循环外稍微调大 HOME_DURATION 以确保末尾停稳 ---
HOME_DURATION = 0.2  # 到末位姿时间 
q_at_end_fixed = None 

while not glfw.window_should_close(window):
    writing_finished_time = cum_time[-1]

    if data.time < writing_finished_time:
        # --- 1. 写字与提笔切换阶段 ---
        for _ in range(40):
            if data.time >= writing_finished_time: break
            idx = np.searchsorted(cum_time, data.time, side='right') - 1
            p0, p1, dur, is_writing = full_plan[np.clip(idx, 0, len(full_plan)-1)]
            tau = np.clip((data.time - cum_time[idx])/dur, 0, 1)
            
            # 使用 S 曲线插值使提笔落笔更平滑
            target_pos = p0 + (p1 - p0) * (tau**2 * (3 - 2 * tau))
            
            # 朝向计算
            z_dir = target_pos - SPHERE_CENTER; z_dir /= np.linalg.norm(z_dir)
            y_tmp = np.array([0, 0, 1]); x_dir = np.cross(y_tmp, z_dir); x_dir /= np.linalg.norm(x_dir)
            y_dir = np.cross(z_dir, x_dir); target_quat = np.zeros(4)
            mj.mju_mat2Quat(target_quat, np.column_stack([x_dir, y_dir, z_dir]).flatten())
            
            old_q = data.qpos.copy()
            dq = IK_controller(model, data, target_pos, target_quat, sid)
            
            # --- 核心改进：如果是提笔状态 (is_writing=False)，加大步长系数 ---
            step_gain = 0.95 if not is_writing else 0.6 
            data.qpos[:6] += dq[:6] * step_gain
            
            # 这里的 10.0 是为了平滑速度，提笔时可以稍微减小以提高响应
            vel_smooth = 5.0 if not is_writing else 10.0
            data.qvel[:6] = (data.qpos[:6] - old_q[:6]) / (model.opt.timestep * vel_smooth)
            
            mj.mj_step(model, data)

            if is_writing: 
                # 记录末端轨迹点用于实时显示
                traj_points.append(data.site_xpos[sid].copy())
                # 新增：记录 6 个关节的当前角度 (qpos) 和 时间，用于最后绘图
                joint_data_log.append(data.qpos[:6].copy())
                time_log.append(data.time)
            
            # 仅在书写状态记录轨迹 
            if is_writing: 
                traj_points.append(data.site_xpos[sid].copy())
        
        q_at_end_fixed = data.qpos[:6].copy()

    elif data.time < (writing_finished_time + HOME_DURATION):
        # --- 2. 快速回位阶段 ---
        if q_at_end_fixed is None: q_at_end_fixed = data.qpos[:6].copy()
        tau = np.clip((data.time - writing_finished_time) / HOME_DURATION, 0, 1)
        smooth_tau = tau**2 * (3 - 2 * tau)
        
        data.qpos[:6] = q_at_end_fixed + smooth_tau * (Q_HOME - q_at_end_fixed)
        data.qvel[:] = 0 
        mj.mj_step(model, data)

    else:
        # --- 3. 稳定锁死 ---
        data.qpos[:6] = Q_HOME
        data.qvel[:] = 0
        mj.mj_forward(model, data)

    # --- 后续渲染逻辑保持不变 ---

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
# ==========================================
# 新增：绘制关节状态曲线
# ==========================================
if joint_data_log:
    joint_data_log = np.array(joint_data_log)
    time_log = np.array(time_log)

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(time_log, joint_data_log[:, i], label=f'Joint {i+1}')

    plt.title('Joint States During Writing (Chinese Family Name)')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (rad)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 自动保存图片，方便放入报告和 PPT
    plt.savefig("joint_states_plot.png")
    print("Joint states plot saved as joint_states_plot.png")
    
    plt.show()