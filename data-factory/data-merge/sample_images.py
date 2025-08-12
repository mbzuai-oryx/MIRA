import os
import random
from PIL import Image
import math

def create_image_collage(input_folder, output_path, sample_size=100, aspect_ratio=(16, 9)):
    # 获取文件夹内所有图片文件
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 如果图片数量少于sample_size，就用全部图片
    sample_size = min(sample_size, len(image_files))
    if sample_size == 0:
        print("文件夹中没有找到图片！")
        return
    
    # 随机抽取图片
    selected_images = random.sample(image_files, sample_size)
    
    # 计算网格布局
    total_images = len(selected_images)
    aspect_width, aspect_height = aspect_ratio
    # 估算每行每列的大约图片数
    cols = int(math.sqrt(total_images * aspect_width / aspect_height))
    rows = math.ceil(total_images / cols)
    
    # 加载所有图片并获取最大尺寸
    images = []
    max_width = 0
    max_height = 0
    
    for img_file in selected_images:
        try:
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
        except Exception as e:
            print(f"无法加载图片 {img_file}: {e}")
            continue
    
    if not images:
        print("没有成功加载任何图片！")
        return
    
    # 计算输出图片的尺寸
    output_width = max_width * cols
    output_height = max_height * rows
    
    # 创建空白画布
    collage = Image.new('RGB', (output_width, output_height), (255, 255, 255))
    
    # 将图片粘贴到画布上
    for idx, img in enumerate(images):
        # 计算当前图片的位置
        row = idx // cols
        col = idx % cols
        
        # 调整图片大小以适应格子
        img_resized = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
        
        # 计算粘贴位置
        x = col * max_width
        y = row * max_height
        
        # 粘贴图片
        collage.paste(img_resized, (x, y))
    
    # 调整最终图片到16:9比例
    target_width = 1920  # 可以调整这个值来改变输出分辨率
    target_height = int(target_width * aspect_height / aspect_width)
    final_collage = collage.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # 保存结果
    final_collage.save(output_path, quality=95)
    print(f"拼接图已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 设置输入文件夹和输出路径
    input_folder = "/Users/moonshot/Documents/清洗MRAG/it_imfiles"  # 替换为你的图片文件夹路径
    output_path = "collage_output.jpg"       # 输出文件名
    
    # 创建拼接图
    create_image_collage(input_folder, output_path, sample_size=104)