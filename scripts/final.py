import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os


# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


# --- 核心改进：高稳定性 IK 控制器 ---
def IK_controller(model, data, X_ref, q_pos):
    site_id = 0  # 末端 site 的索引
    
    current_pos = data.site_xpos[site_id]
    current_mat = data.site_xmat[site_id].reshape(3, 3)

    # 目标旋转：笔尖垂直向下
    R_ref = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    # 1. 计算位置和旋转误差
    pos_err = X_ref - current_pos
    # 旋转误差：使用叉乘法计算当前姿态与目标姿态的偏差
    rot_err = 0.5 * (
        np.cross(current_mat[:, 0], R_ref[:, 0]) +
        np.cross(current_mat[:, 1], R_ref[:, 1]) +
        np.cross(current_mat[:, 2], R_ref[:, 2])
    )
    
    full_err = np.concatenate([pos_err, rot_err])
    err_norm = np.linalg.norm(full_err)
    
    # 2. 极其微小的死区，防止静止时的数值震荡
    if err_norm < 1e-4:
        return q_pos
    
    # 3. 计算 site 雅可比
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp, jacr])

    # 4. 阻尼最小二乘法 (Damped Least Squares)
    # 较大的 damping (0.15) 能有效吸收奇点附近的无穷大速度，解决“抽风”
    damping = 0.15
    I = np.eye(6)
    # 公式: dq = J^T * (J*J^T + λ^2*I)^-1 * error
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * I, full_err)

    # 5. 步长强制截断 (Clamping)
    # 限制单步关节最大位移（弧度），防止物理引擎因位置突跳崩溃
    max_step = 0.02 
    actual_step_norm = np.linalg.norm(dq)
    if actual_step_norm > max_step:
        dq = dq * (max_step / actual_step_norm)

    # 6. 控制增益 (Alpha)
    alpha = 0.5
    return q_pos + dq * alpha

    
def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)


############################
# 模型加载与环境配置
############################
xml_path = 'D:/Final Project of Robowriter/models/universal_robots_ur5e/scene.xml'
simend = 180.0 

script_dir = os.path.dirname(__file__)
xml_full = os.path.abspath(os.path.join(script_dir, xml_path)) 

if not os.path.exists(xml_full):
    print(f"ERROR: XML path not found at {xml_full}")
    # 如果报错，请在这里手动填入你的绝对路径
    # xml_full = "/YOUR/ABSOLUTE/PATH/TO/scene.xml"

model = mj.MjModel.from_xml_path(xml_full)
data = mj.MjData(model)

# 提高物理引擎步进精度
model.opt.timestep = 0.001 

cam = mj.MjvCamera()
opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(1600, 900, "Stable Vertical Writing", None, None)
glfw.make_context_current(window)
# --- 在这里注册回调函数 ---
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

mj.mjv_defaultCamera(cam)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 初始位姿 (确保起始状态接近垂直，避开奇异姿态)
data.qpos[:] = np.array([-1.6, -1.3, 2.1, -2.6, -1.57, 0])
mj.mj_forward(model, data)

cam.lookat = np.array([0.0, 0.32, 0.05])
cam.distance = 1.2

############################
# 轨迹规划 (写 "刘" - 简化版)
############################
z_write = 0.1  # 严格符合项目要求
z_lift = 0.15  # 抬笔高度

# 定义“刘”字各个关键点 (居中放置)
# 左半部分 (文)
p_dot_s = np.array([-0.2237490060074, 0.5344489582474, z_write])  # 点
p_dot_e = np.array([-0.2120797753758, 0.5173097757573, z_write])

p_heng_s = np.array([-0.2480249172111, 0.5098709538317, z_write]) # 横
p_heng_e = np.array([-0.1793630381269, 0.523846380548, z_write])

p_pie_s = np.array([-0.1940682161563, 0.520853291214, z_write])  # 撇
p_pie_e = np.array([-0.2352647449919, 0.458830264955, z_write])

p_dian_s = np.array([-0.2291884725066, 0.4995412906067, z_write]) # 捺/点
p_dian_e = np.array([-0.1890850741034, 0.478274336908, z_write])

# 右半部分 (立刀旁)
p_shu_s = np.array([-0.168091834841, 0.5093199018429, z_write])   # 短竖
p_shu_e = np.array([-0.1684257476533, 0.4855658638904, z_write])

p_shugou_s = np.array([-0.1549512786255, 0.5434787351868, z_write]) # 竖钩
p_shugou_m = np.array([-0.1537336222181, 0.4483883823397, z_write])
p_shugou_e = np.array([-0.168091834841, 0.4593199018429, z_write])

plan = []

def add_stroke(p0, p1, dur, contact):
    plan.append((p0, p1, dur, contact))

# --- 1. 写点 ---
add_stroke(p_dot_s + [0, 0, 0.05], p_dot_s, 0.5, False) # 下笔
add_stroke(p_dot_s, p_dot_e, 0.5, True)                # 笔画
add_stroke(p_dot_e, p_dot_e + [0, 0, 0.05], 0.3, False) # 抬笔

# --- 2. 写横 ---
add_stroke(p_dot_e + [0, 0, 0.05], p_heng_s + [0, 0, 0.05], 0.5, False) # 移动
add_stroke(p_heng_s + [0, 0, 0.05], p_heng_s, 0.3, False)
add_stroke(p_heng_s, p_heng_e, 1.0, True)
add_stroke(p_heng_e, p_heng_e + [0, 0, 0.05], 0.3, False)

# --- 3. 写撇 ---
add_stroke(p_heng_e + [0, 0, 0.05], p_pie_s + [0, 0, 0.05], 0.5, False)
add_stroke(p_pie_s + [0, 0, 0.05], p_pie_s, 0.3, False)
add_stroke(p_pie_s, p_pie_e, 0.8, True)
add_stroke(p_pie_e, p_pie_e + [0, 0, 0.05], 0.3, False)

# --- 4. 写点 (右下) ---
add_stroke(p_pie_e + [0, 0, 0.05], p_dian_s + [0, 0, 0.05], 0.5, False)
add_stroke(p_dian_s + [0, 0, 0.05], p_dian_s, 0.3, False)
add_stroke(p_dian_s, p_dian_e, 0.6, True)
add_stroke(p_dian_e, p_dian_e + [0, 0, 0.05], 0.3, False)

# --- 5. 写短竖 ---
add_stroke(p_dian_e + [0, 0, 0.05], p_shu_s + [0, 0, 0.05], 0.7, False)
add_stroke(p_shu_s + [0, 0, 0.05], p_shu_s, 0.3, False)
add_stroke(p_shu_s, p_shu_e, 0.7, True)
add_stroke(p_shu_e, p_shu_e + [0, 0, 0.05], 0.3, False)

# --- 6. 写竖钩 ---
add_stroke(p_shu_e + [0, 0, 0.05], p_shugou_s + [0, 0, 0.05], 0.5, False)
add_stroke(p_shugou_s + [0, 0, 0.05], p_shugou_s, 0.3, False)
add_stroke(p_shugou_s, p_shugou_m, 1.2, True) # 竖
add_stroke(p_shugou_m, p_shugou_e, 0.4, True) # 钩
add_stroke(p_shugou_e, p_shugou_e + [0, 0, 0.1], 1.0, False) # 最终抬笔

cum = np.cumsum([0] + [p[2] for p in plan])
total_time = cum[-1]
traj_points = []

############################
# 主循环
############################
while not glfw.window_should_close(window):
    sim_time = data.time
    
    if sim_time < simend:
        # 高频控制循环 (200Hz)
        while data.time - sim_time < 0.005:
            s = min(data.time, total_time)
            idx = np.searchsorted(cum, s, side='right') - 1
            idx = np.clip(idx, 0, len(plan)-1)
            
            p0, p1, dur, contact = plan[idx]
            tau = (s - cum[idx]) / max(dur, 1e-6)
            tau = np.clip(tau, 0.0, 1.0)
            X_ref = p0 + tau * (p1 - p0)
            
            # 1. 计算 IK 目标位置
            prev_qpos = data.qpos.copy()
            new_qpos = IK_controller(model, data, X_ref, prev_qpos)
            
            # 2. 关键同步：更新 qpos 的同时手动计算 qvel
            # 物理引擎如果只看到 qpos 突变而没有速度，会产生巨大的虚假穿透力导致抽风
            data.qvel[:] = (new_qpos - prev_qpos) / model.opt.timestep
            data.qpos[:] = new_qpos
            
            # 3. 物理步进
            mj.mj_step(model, data)
            
            if contact:
                traj_points.append(data.site_xpos[0].copy())
    
    # 渲染
    viewport = mj.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    
    # 绘制轨迹
    for p in traj_points[::10]: # 抽样绘制，减轻渲染压力
        if scene.ngeom < scene.maxgeom:
            g = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, [0.003,0,0], p, np.eye(3).flatten(), [0, 0.8, 1, 1])
    
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()