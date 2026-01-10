# -*- coding: GBK -*-

from PIL import Image, ImageDraw, ImageFont
import os
import platform

def get_system_fonts():
    """
    根据操作系统返回中文字体路径列表
    """
    system_name = platform.system()
    
    if system_name == "Windows":
        fonts_dir = "C:\\Windows\\Fonts\\"
        return [
            fonts_dir + "msyh.ttc",  # 微软雅黑
            fonts_dir + "simsun.ttc",  # 宋体
            fonts_dir + "simhei.ttf",  # 黑体
            fonts_dir + "simkai.ttf",  # 楷体
            fonts_dir + "Deng.ttf",  # 等线
            fonts_dir + "simfang.ttf",  # 仿宋
        ]
    elif system_name == "Darwin":  # macOS
        fonts_dir = "/System/Library/Fonts/"
        return [
            fonts_dir + "PingFang.ttc",  # 苹方
            fonts_dir + "STHeiti Light.ttc",  # 华文黑体
            fonts_dir + "STSong.ttc",  # 华文宋体
            "/Library/Fonts/Arial Unicode.ttf",  # Arial Unicode
        ]
    else:  # Linux
        fonts_dir = "/usr/share/fonts/"
        return [
            fonts_dir + "truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            fonts_dir + "opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Droid Sans
        ]

def create_chinese_image_auto(text, output_path, font_size=80):
    """
    自动检测系统并选择合适的字体
    """
    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取系统字体列表
    font_paths = get_system_fonts()
    
    # 添加一些通用字体尝试
    font_paths.extend([
        "simsun.ttc",  # 常用名称
        "msyh.ttc",  # 常用名称
        "arialuni.ttf",  # Arial Unicode
    ])
    
    font = None
    selected_font = ""
    
    # 尝试所有字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                selected_font = font_path
                print(f"成功加载字体: {os.path.basename(font_path)}")
                break
            except:
                continue
    
    # 如果所有字体都失败，使用默认字体
    if font is None:
        print("警告：未找到中文字体，将使用默认字体（可能显示方框）")
        font = ImageFont.load_default()
    
    # 计算文本大小
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    try:
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # 估算大小
        text_width = len(text) * font_size
        text_height = font_size * 1.5
    
    # 创建图片
    margin = 50
    img_width = text_width + 2 * margin
    img_height = text_height + 2 * margin
    
    image = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 绘制文本，水平与垂直居中
    text_x = int((img_width - text_width) / 2)
    text_y = int((img_height - text_height) / 2)
    draw.text((text_x, text_y), text, font=font, fill='black')
    
    # 可选：添加边框
    draw.rectangle([(5, 5), (img_width-6, img_height-6)], outline="gray", width=2)
    
    # 保存图片
    image.save(output_path, 'JPEG', quality=95)
    print(f"图片已保存到: {output_path}")
    print(f"图片尺寸: {img_width} × {img_height} 像素")
    
    return image

def main():
    # 目标路径
    output_path = r'C:\Users\mhysx\Downloads\main\SIST_SI100B_RoboWriter-main\scripts\lec5_planning\character.jpg'
    
    print("=" * 50)
    print("中文文本转图片生成器")
    print("=" * 50)
    print(f"检测到操作系统: {platform.system()} {platform.release()}")
    
    user_text = input("\n请输入要生成图片的中文文本: ")
    
    if not user_text.strip():
        print("错误：输入不能为空！")
        return
    
    try:
        font_size = int(input("请输入字体大小 (默认80): ") or "80")
    except:
        font_size = 80
    
    # 创建图片
    image = create_chinese_image_auto(user_text, output_path, font_size)
    
    # 询问是否显示图片
    show = input("\n是否显示生成的图片? (y/n): ").lower()
    if show == 'y':
        image.show()
    
    print("\n完成！")

if __name__ == "__main__":
    main()