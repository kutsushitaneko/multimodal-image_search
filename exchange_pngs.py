import os
import shutil

def exchange_pngs():
    pngs_folder = 'pngs'
    images_folder = 'images'

    # pngsフォルダー内のすべてのPNGファイルをループ
    for png_file in os.listdir(pngs_folder):
        if png_file.lower().endswith('.png'):
            png_path = os.path.join(pngs_folder, png_file)
            base_name = os.path.splitext(png_file)[0]
            
            # 対応するJPEGファイルのパス
            jpeg_path = os.path.join(images_folder, base_name + '.jpg')
            
            # JPEGファイルが存在する場合、削除
            if os.path.exists(jpeg_path):
                os.remove(jpeg_path)
            
            # PNGファイルをimagesフォルダーにコピー
            shutil.copy2(png_path, os.path.join(images_folder, png_file))

    print("処理が完了しました。")

if __name__ == "__main__":
    exchange_pngs()