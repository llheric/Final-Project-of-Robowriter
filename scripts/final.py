import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt  # 绘图库

# ==========================================
# 1. 核心参数与配置
# ==========================================
SLOWDOWN_FACTOR = 0.3    # 全局速度调节因子
RENDER_BATCH_SIZE = 100   # 每帧渲染的仿真步数
DRAW_SKIP_STEP = 1       # 轨迹点绘制间隔
MAX_GEOM_LIMIT = 200000   # 最大几何体数量

USE_BEZIER = True      # 是否使用贝塞尔曲线插值，False 则使用线性插值 
USE_SMOOTH_STEP = True  
DYNAMIC_THICKNESS = True 

X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = 0.1, 0.6
Z_WRITE = 0.1      
PADDING = 0.5  # 字体边距比例          

Q_STOP = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0])
STOP_MOVE_DURATION = 1.0  

button_left, button_right = False, False
lastx, lasty = 0, 0

# ==========================================
# 2. 插值函数库
# ==========================================
def get_linear_plan(pts, step_dur=0.01):
    segments = []
    for i in range(len(pts) - 1):
        segments.append((pts[i], pts[i+1], step_dur * SLOWDOWN_FACTOR, True))
    return segments

def get_bezier_plan(pts, step_dur=0.01, sub_steps=5):
    def bezier_calc(p0, p1, p2, t):
        return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    segments = []
    if len(pts) < 3: return get_linear_plan(pts, step_dur) 
    for i in range(len(pts) - 2):
        p0, p1, p2 = pts[i], pts[i+1], pts[i+2]
        for j in range(sub_steps):
            t0, t1 = j/sub_steps, (j+1)/sub_steps
            segments.append((bezier_calc(p0, p1, p2, t0), 
                             bezier_calc(p0, p1, p2, t1), 
                             (step_dur/sub_steps) * SLOWDOWN_FACTOR, True))
    return segments

def s_curve_interp(t):
    return t * t * (3 - 2 * t)

# ==========================================
# 3. 辅助功能 (视角、IK)
# ==========================================
def mouse_button(window, button, act, mods):
    global button_left, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

def mouse_move(window, xpos, ypos):
    global lastx, lasty
    dx, dy = xpos - lastx, ypos - lasty
    lastx, lasty = xpos, ypos
    if not (button_left or button_right): return
    width, height = glfw.get_window_size(window)
    action = mj.mjtMouse.mjMOUSE_ROTATE_V if button_left else mj.mjtMouse.mjMOUSE_MOVE_V
    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scene, cam)

def IK_controller(model, data, X_ref, q_pos):
    site_id = 0
    cur_pos = data.site_xpos[site_id]
    cur_mat = data.site_xmat[site_id].reshape(3, 3)
    R_ref = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) 
    err = np.concatenate([X_ref - cur_pos, 0.5*(np.cross(cur_mat[:,0], R_ref[:,0])+np.cross(cur_mat[:,1], R_ref[:,1])+np.cross(cur_mat[:,2], R_ref[:,2]))])
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp, jacr])
    dq = J.T @ np.linalg.solve(J @ J.T + 0.15**2 * np.eye(6), err)
    return q_pos + dq * 0.5

# ==========================================
# 4. 轨迹规划与排序
# ==========================================
stroke_data_path = "stroke_data.npy"
if not os.path.exists(stroke_data_path):
    print("错误: 找不到数据文件"); exit()

strokes = np.load(stroke_data_path, allow_pickle=True)

def get_stroke_center(s):
    pts = np.array(s)
    return np.mean(pts[:, 0]), np.mean(pts[:, 1])

indexed_strokes = []
for s in strokes:
    if len(s) < 2: continue
    cx, cy = get_stroke_center(s)
    indexed_strokes.append({'s': s, 'cx': cx, 'cy': cy})

row_threshold = 0.05 
strokes_sorted = sorted(indexed_strokes, key=lambda k: (round(k['cy'] / row_threshold), k['cx']))
strokes = [item['s'] for item in strokes_sorted]

rect_center = np.array([(X_MAX+X_MIN)/2, (Y_MAX+Y_MIN)/2])+np.array([0, 0.08])  # 调整整体位置
all_pts_flat = np.vstack([s for s in strokes])
raw_min, raw_max = all_pts_flat.min(axis=0), all_pts_flat.max(axis=0)
raw_center = (raw_max + raw_min) / 2.0
auto_scale = min((X_MAX-X_MIN)/(raw_max[0]-raw_min[0]), (Y_MAX-Y_MIN)/(raw_max[1]-raw_min[1])) * PADDING

image_plan = []
for s in strokes:
    pts = []
    
    Y_OFFSET = 0.00  # 字体向上移动的距离（单位：米）。0.05 表示向上移动 5 厘米

    for p in s:
        nx = rect_center[0] + (p[0] - raw_center[0]) * auto_scale
        ny = rect_center[1] - (p[1] - raw_center[1]) * auto_scale + Y_OFFSET # 加上偏移量
        
        # 确保加上偏移后不会超出定义的 Y_MAX
        pts.append(np.array([np.clip(nx, X_MIN, X_MAX), 
                             np.clip(ny, Y_MIN, Y_MAX), 
                             Z_WRITE]))
    
    p_start = pts[0]
    image_plan.append((p_start + [0, 0, 0.05], p_start + [0, 0, 0.05], 0.4 * SLOWDOWN_FACTOR, False))
    image_plan.append((p_start + [0, 0, 0.05], p_start, 0.2 * SLOWDOWN_FACTOR, False))
    
    if USE_BEZIER:
        image_plan.extend(get_bezier_plan(pts))
    else:
        image_plan.extend(get_linear_plan(pts))
    
    image_plan.append((pts[-1], pts[-1] + [0, 0, 0.05], 0.2 * SLOWDOWN_FACTOR, False))

plan = image_plan
cum = np.cumsum([0] + [p[2] for p in plan])
writing_time = cum[-1]
final_time = writing_time + STOP_MOVE_DURATION

# ==========================================
# 5. MuJoCo 初始化
# ==========================================
xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/universal_robots_ur5e/scene.xml'))
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
model.opt.timestep = 0.002

glfw.init()
window = glfw.create_window(1400, 900, "Ordered Multi-Interpolation", None, None)
glfw.make_context_current(window)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

scene = mj.MjvScene(model, maxgeom=MAX_GEOM_LIMIT)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
# ==========================================
# 5. MuJoCo 初始化 (视角调整部分)
# ==========================================


cam = mj.MjvCamera()
mj.mjv_defaultCamera(cam)

# 设置目标点为写字区域的中心
cam.lookat = [0, 0.35, 0] 

# 核心设置：竖直向下看
cam.elevation = -90.0  # 仰角设为 -90 度
cam.azimuth = 90.0     # 方位角设为 90 度（可以根据需要调整 0, 90, 180, 270）
cam.distance = 1.0     # 相机距离目标点的距离，根据字迹大小调整
data.qpos[:] = [-1.57, -1.57, 1.57, -1.57, -1.57, 0]
traj_points = []
q_at_end = data.qpos.copy()
is_finished = False

# --- 新增：关节状态记录容器 ---
joint_data_history = []
time_history = []
plot_shown = False 

# --- 6. 主循环 ---
while not glfw.window_should_close(window):
    s = data.time
    if not is_finished:
        # --- 新增：记录当前时刻的关节角度和时间 ---
        joint_data_history.append(data.qpos.copy())
        time_history.append(s)

        if s < writing_time:
            for _ in range(int(RENDER_BATCH_SIZE)):
                if data.time >= writing_time: break
                idx = np.searchsorted(cum, data.time, side='right') - 1
                idx = np.clip(idx, 0, len(plan)-1)
                p0, p1, dur, contact = plan[idx]
                tau = np.clip((data.time - cum[idx]) / max(dur, 1e-6), 0, 1)
                if USE_SMOOTH_STEP: tau = s_curve_interp(tau)
                X_ref = p0 + tau * (p1 - p0)
                prev_q = data.qpos.copy()
                data.qpos[:] = IK_controller(model, data, X_ref, prev_q)
                data.qvel[:] = (data.qpos - prev_q) / model.opt.timestep
                mj.mj_step(model, data)
                if contact and abs(data.site_xpos[0][2] - Z_WRITE) < 0.005:
                    if int(data.time * 1000) % 3 == 0:
                        vel = np.linalg.norm(data.qvel)
                        radius = max(0.0015, 0.004 - vel * 0.0006) if DYNAMIC_THICKNESS else 0.002
                        traj_points.append((data.site_xpos[0].copy(), radius))
            q_at_end = data.qpos.copy()
        elif s < final_time:
            for _ in range(int(RENDER_BATCH_SIZE)):
                tau_stop = np.clip((data.time - writing_time) / STOP_MOVE_DURATION, 0, 1)
                tau_smooth = s_curve_interp(tau_stop)
                data.qpos[:] = q_at_end + tau_smooth * (Q_STOP - q_at_end)
                data.qvel[:] = 0
                mj.mj_step(model, data)
        else:
            is_finished = True
    else:
        # --- 新增：当任务结束时，绘制曲线 ---
        if not plot_shown:
            joint_history_np = np.array(joint_data_history)
            plt.figure(figsize=(10, 6))
            for i in range(model.nq):
                plt.plot(time_history, joint_history_np[:, i], label=f'Joint {i+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Angle (rad)')
            plt.title('Joint States Over Time')
            plt.legend()
            plt.grid(True)
            plt.show() # 注意：这会阻塞程序直到关闭图片窗口
            plot_shown = True

        data.qpos[:] = Q_STOP
        data.qvel[:] = 0
        mj.mj_forward(model, data)

    # 渲染
    viewport = mj.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, 3, scene)
    for p, r in traj_points[::DRAW_SKIP_STEP]:
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(scene.geoms[scene.ngeom], 2, [r,0,0], p, np.eye(3).flatten(), [0, 0.5, 1, 1])
            scene.ngeom += 1
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window); glfw.poll_events()

glfw.terminate()