import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心参数与配置
# ==========================================
SLOWDOWN_FACTOR = 0.3
RENDER_BATCH_SIZE = 100
DRAW_SKIP_STEP = 1
MAX_GEOM_LIMIT = 200000

USE_BEZIER = True
USE_SMOOTH_STEP = True
DYNAMIC_THICKNESS = True

# 球面参数定义
CX, CY, CZ = 0.0, 0.35, 1.3
R = 1.3
R_SQ = 1.69

# 限制平面映射范围以满足 0 <= z <= 0.1
# 根据方程，当 z=0.1 时，水平半径约为 0.5
X_MIN, X_MAX = -0.35, 0.35
Y_MIN, Y_MAX = 0.0, 0.7  # 相对于 CY=0.35 的偏移
PADDING = 0.7 

Q_STOP = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0])
STOP_MOVE_DURATION = 1.0

# ==========================================
# 2. 球面投影与法线计算
# ==========================================
def get_spherical_coords(x, y):
    """根据 x, y 计算球面上对应的 z 和法线方向"""
    dx = x - CX
    dy = y - CY
    dist_sq = dx**2 + dy**2
    
    # 确保在球内，防止开根号出现负数
    if dist_sq > R_SQ - 0.01:
        dist_sq = R_SQ - 0.01
        
    # 计算 z (取下半球)
    z = CZ - np.sqrt(R_SQ - dist_sq)
    
    # 计算法线方向 (从接触点指向球心)
    pos = np.array([x, y, z])
    center = np.array([CX, CY, CZ])
    normal = (center - pos) / np.linalg.norm(center - pos)
    
    return pos, normal

# ==========================================
# 3. 改进的 IK 控制器 (支持方向控制)
# ==========================================
def IK_controller_v2(model, data, X_ref, normal_ref, q_pos):
    site_id = 0
    cur_pos = data.site_xpos[site_id]
    cur_mat = data.site_xmat[site_id].reshape(3, 3)
    
    # 构建目标旋转矩阵：让笔尖 Z 轴指向 normal_ref (即球心方向)
    # 这里简写：依然采用原本的参考坐标系，但加入微调，或者维持 R_ref 确保稳定
    # 为了稳定，我们让笔尖大致垂直向上对准球心
    z_axis = normal_ref
    x_axis = np.array([1, 0, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    R_ref = np.column_stack([x_axis, y_axis, z_axis])

    err_pos = X_ref - cur_pos
    err_rot = 0.5 * (np.cross(cur_mat[:,0], R_ref[:,0]) + 
                     np.cross(cur_mat[:,1], R_ref[:,1]) + 
                     np.cross(cur_mat[:,2], R_ref[:,2]))
    
    err = np.concatenate([err_pos, err_rot])
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp, jacr])
    dq = J.T @ np.linalg.solve(J @ J.T + 0.1**2 * np.eye(6), err)
    return q_pos + dq * 0.5

# ==========================================
# 4. 轨迹规划
# ==========================================
stroke_data_path = "stroke_data.npy"
if not os.path.exists(stroke_data_path):
    print("Error: stroke_data.npy not found"); exit()

strokes = np.load(stroke_data_path, allow_pickle=True)
all_pts_flat = np.vstack([s for s in strokes])
raw_min, raw_max = all_pts_flat.min(axis=0), all_pts_flat.max(axis=0)
raw_center = (raw_max + raw_min) / 2.0
auto_scale = min((X_MAX-X_MIN)/(raw_max[0]-raw_min[0]), (Y_MAX-Y_MIN)/(raw_max[1]-raw_min[1])) * PADDING

image_plan = []
for s in strokes:
    pts_and_normals = []
    for p in s:
        nx = CX + (p[0] - raw_center[0]) * auto_scale
        ny = CY - (p[1] - raw_center[1]) * auto_scale
        pos, normal = get_spherical_coords(nx, ny)
        pts_and_normals.append((pos, normal))
    
    # 规划：提笔、下笔、写字、提笔
    p_start, n_start = pts_and_normals[0]
    lift_vec = n_start * -0.04 # 沿法线反方向（向球心内）提笔
    
    image_plan.append((p_start + lift_vec, n_start, 0.4 * SLOWDOWN_FACTOR, False))
    image_plan.append((p_start, n_start, 0.2 * SLOWDOWN_FACTOR, True))
    
    for i in range(len(pts_and_normals)-1):
        p0, n0 = pts_and_normals[i]
        p1, n1 = pts_and_normals[i+1]
        image_plan.append((p1, n1, 0.02 * SLOWDOWN_FACTOR, True))
    
    p_end, n_end = pts_and_normals[-1]
    image_plan.append((p_end + lift_vec, n_end, 0.2 * SLOWDOWN_FACTOR, False))

cum = np.cumsum([0] + [p[2] for p in image_plan])
writing_time = cum[-1]

# ==========================================
# 5. MuJoCo 运行
# ==========================================
xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/universal_robots_ur5e/scene.xml'))
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

glfw.init()
window = glfw.create_window(1200, 800, "Spherical Inner Surface Writing", None, None)
glfw.make_context_current(window)

scene = mj.MjvScene(model, maxgeom=MAX_GEOM_LIMIT)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
cam = mj.MjvCamera()
mj.mjv_defaultCamera(cam)
cam.lookat = [CX, CY, 0.1]
cam.distance = 1.2
cam.elevation = -30

traj_points = []
is_finished = False

while not glfw.window_should_close(window):
    if not is_finished:
        if data.time < writing_time:
            for _ in range(RENDER_BATCH_SIZE):
                t = data.time
                idx = np.searchsorted(cum, t, side='right') - 1
                idx = np.clip(idx, 0, len(image_plan)-1)
                
                target_pos, target_normal, dur, is_writing = image_plan[idx]
                
                # 执行 IK
                data.qpos[:] = IK_controller_v2(model, data, target_pos, target_normal, data.qpos.copy())
                mj.mj_step(model, data)
                
                if is_writing:
                    # 记录轨迹（红色点）
                    if int(t * 1000) % 2 == 0:
                        traj_points.append(data.site_xpos[0].copy())
        else:
            is_finished = True

    # 渲染
    viewport = mj.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
    mj.mjv_updateScene(model, data, mj.MjvOption(), None, cam, 3, scene)
    
    # 绘制球面轨迹
    for p in traj_points:
        if scene.ngeom < scene.maxgeom:
            mj.mjv_initGeom(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, np.eye(3).flatten(), [1, 0, 0, 1])
            scene.ngeom += 1
            
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()