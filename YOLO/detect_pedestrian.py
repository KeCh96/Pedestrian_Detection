from PIL import Image, ImageFont, ImageDraw
from model import yolo_eval, YOLO
from matplotlib import pyplot as plt
import time
from glob import glob
import os
import argparse
from mAP import eval_mAP
import pandas as pd



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
result_dir = './evaluate/result_evaluate/'
pred_dir = './evaluate/predicted/'
gt_dir = './evaluate/ground_truth/'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, help="give dataset dir path by '--dir' !!!")
parser.add_argument('--gt_csv', type=str, help="give ground truth file name by '--gt_csv' !!!")
parser.add_argument('--result_csv', type=str, help="give result output filename by '--result_csv' !!!")
args = parser.parse_args()

if __name__ == '__main__':
    image_paths_list = glob(args.image_dir + '*')
    print('Find {} images.'.format(len(image_paths_list)))

    yolo = YOLO()
    yolo.detect_on_set(image_paths_list = image_paths_list, output_csv_name = args.result_csv, object='person', save_animation=True)
    yolo.close_session()

    eval_mAP(
        pred_file_path= pred_dir + args.result_csv,
        gt_file_path= gt_dir + args.gt_csv,
        output_dir= result_dir,
        draw_plot= True
    )