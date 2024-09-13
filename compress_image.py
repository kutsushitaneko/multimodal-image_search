import os
from PIL import Image

def compress_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{filename.split('.')[0]}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            with Image.open(input_path) as img:
                # リサイズ処理を追加
                if max(img.size) > 512:
                    img.thumbnail((512, 512), Image.LANCZOS)
                img.convert('RGB').save(output_path, 'JPEG', optimize=True, quality=65)

            print(f"{filename} を圧縮・リサイズして {output_filename} として保存しました。")

if __name__ == "__main__":
    input_folder = "samples"
    output_folder = "images"
    compress_images(input_folder, output_folder)