from PIL import Image
import sys

def extract_text_chunks(png_file):
    try:
        with Image.open(png_file) as img:
            text_chunks = {}
            for key, value in img.info.items():
                if isinstance(value, str):
                    text_chunks[key] = value
            return text_chunks
    except IOError:
        print(f"Error: Unable to open file {png_file}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <png_file>")
        sys.exit(1)

    png_file = sys.argv[1]
    chunks = extract_text_chunks(png_file)

    if chunks:
        print("Extracted text chunks:")
        for key, value in chunks.items():
            print("-----------------------------------------")
            print(f"{key}:\n{value}")
            print("-----------------------------------------")
    else:
        print("No text chunks found or error occurred.")

if __name__ == "__main__":
    main()
