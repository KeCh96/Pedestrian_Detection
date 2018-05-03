from PIL import Image, ImageFont, ImageDraw
from model import yolo_eval, YOLO
from matplotlib import pyplot as plt
import time
from glob import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'




if __name__ == '__main__':
    image_list = glob('./test_image/*')
    print(image_list)
    yolo = YOLO()
    for img in image_list:
        # image = input('Input image filename:')
        image = Image.open(img)
        image = yolo.detect_image(image)
        image.save('./result/' + img.split('/')[-1])
    yolo.close_session()