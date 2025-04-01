import cv2

img = cv2.imread('/data/train_GAN/texture.jpeg', cv2.IMREAD_COLOR)
if img is None:
    print("cv2 couldn't load the image")
else:
    print("Image loaded successfully:", img.shape)
    
from PIL import features
print(features.check("avif"))