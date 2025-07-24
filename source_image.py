from PIL import Image

def create_white_image(width=512, height=512, output_path='example.png'):
    """
    此函数用于创建指定尺寸的白色图像。

    参数:
    width (int): 图像的宽度，默认值为512。
    height (int): 图像的高度，默认值为512。
    output_path (str): 保存图像的文件路径，默认值为'white_image.png'。

    返回:
    Image: 创建好的白色图像对象。
    """
    # 生成白色图像，'RGB'模式，全255（白色）
    white_image = Image.new('RGB', (width, height), color=(255, 255, 255))
    # 保存图像
    white_image.save(output_path)
    print(f"已成功创建并保存白色图像至 {output_path}")
    return white_image

create_white_image()