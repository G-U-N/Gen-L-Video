from PIL import Image
import os
def compress_gif(input_path, output_path, target_size):
    with Image.open(input_path) as image:
        # 获取GIF的帧
        frames = []
        try:
            while True:
                frames.append(image.copy())
                image.seek(len(frames))  # 移动到下一帧
        except EOFError:
            pass

        # 压缩帧
        compressed_frames = []
        for frame in frames:
            frame.thumbnail((800,800))  # 可选步骤：调整帧的大小
            compressed_frames.append(frame)

        # 保存压缩后的GIF
        compressed_frames[0].save(output_path, save_all=True, append_images=compressed_frames[1:], optimize=True,
                                  duration=image.info['duration'], loop=0)

    # 检查文件大小
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # 转换为MB
    print('压缩后的文件大小为: {:.2f}MB'.format(compressed_size))

    return compressed_size

# 输入和输出文件路径
input_file = 'fight.gif'
output_file = 'compressed.gif'

# 目标文件大小（单位：MB）
target_size = 100

# 执行压缩
compress_gif(input_file, output_file, target_size)
