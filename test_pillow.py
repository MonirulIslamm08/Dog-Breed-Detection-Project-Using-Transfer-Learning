from PIL import Image

img_path = 'D:\\Download\\bulldog56.jpg'  # Replace with your image path
try:
    img = Image.open(img_path)
    img.show()
    print("Pillow is working!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
