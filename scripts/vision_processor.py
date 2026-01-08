import cv2
import numpy as np
import os
import sys
import glob
from skimage.morphology import skeletonize

def extract_strokes_from_image(image_path, save_name="stroke_data.npy"):
    """
    识别图片中的字符，提取骨架，并自动划分笔画
    """
    # 1. 读取并预处理
    if not os.path.exists(image_path):
        print(f"Error: 找不到图片 {image_path}")
        return
    
    # 读取为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 二值化 (假设白底黑字，进行反色处理使字为白色/高亮)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 骨架化 (Skeletonization)
    # 将粗体笔画坍缩为 1 像素宽度的中心线
    skeleton = skeletonize(binary > 0)
    y, x = np.where(skeleton)
    points = np.column_stack([x, y]).astype(float)
    
    if len(points) == 0:
        print("未识别到任何有效笔画")
        return

    # 3. 笔画拓扑分析与排序
    # 我们利用最近邻算法寻找连续点。如果距离突然变大，说明是“下一笔”
    strokes = []
    remaining = points.tolist()
    
    # 初始点
    current_stroke = [remaining.pop(0)]
    jump_threshold = 5.0  # 像素阈值：如果下个点距离超过5像素，则认为需抬笔
    
    while remaining:
        last_pt = current_stroke[-1]
        # 计算剩余点到当前点末尾的距离
        dists = np.linalg.norm(np.array(remaining) - last_pt, axis=1)
        min_idx = np.argmin(dists)
        
        if dists[min_idx] < jump_threshold:
            current_stroke.append(remaining.pop(min_idx))
        else:
            # 保存当前笔画，开始新笔画
            if len(current_stroke) > 3: # 过滤噪点
                strokes.append(np.array(current_stroke))
            current_stroke = [remaining.pop(min_idx)]
            
    # 添加最后一笔
    if len(current_stroke) > 3:
        strokes.append(np.array(current_stroke))

    # 4. 坐标转换 (归一化到 -1 到 1 之间，方便主程序映射)
    processed_strokes = []
    center_x, center_y = np.mean(x), np.mean(y)
    max_dim = max(img.shape)

    for s in strokes:
        # 将像素坐标转为以中心为原点的相对坐标
        s[:, 0] = (s[:, 0] - center_x) / max_dim
        s[:, 1] = (s[:, 1] - center_y) / max_dim
        processed_strokes.append(s)

    # 5. 保存数据
    # 数据结构：一个列表，每个元素是一个 (N, 2) 的笔画坐标矩阵
    np.save(save_name, np.array(processed_strokes, dtype=object))
    print(f"成功识别 {len(processed_strokes)} 条笔画，已保存至 {save_name}")

def _find_candidate_image(default_name="character.jpg"):
    """Try several candidate locations for the image and return the first found path."""
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, default_name),
        os.path.join(script_dir, '..', default_name),
        os.path.join(os.getcwd(), default_name),
        default_name,
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    # try any common image in current working directory
    imgs = glob.glob(os.path.join(os.getcwd(), '*.jpg')) + glob.glob(os.path.join(os.getcwd(), '*.png'))
    if imgs:
        return imgs[0]

    return None


if __name__ == "__main__":
    # Prefer a command-line argument if provided
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = _find_candidate_image('character.jpg')

    if img_path is None:
        print('Error: 找不到图片 character.jpg，也未在当前目录找到任何 JPG/PNG 文件。')
        files = glob.glob(os.path.join(os.getcwd(), '*'))
        print('当前目录下的文件（部分）：')
        for f in files[:50]:
            print('  -', os.path.basename(f))
        print('\n解决方法：')
        print('  1) 把你的图片放到脚本同目录，命名为 character.jpg；')
        print('  2) 或者运行：python vision_processor.py path/to/your_image.jpg')
        sys.exit(1)

    extract_strokes_from_image(img_path)