from model import  YOLO
from glob import glob
import os
import argparse



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
result_dir = './evaluate/result_evaluate/'
pred_dir = './evaluate/predicted/'
gt_dir = './evaluate/ground_truth/'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, help="give dataset dir path by '--image_dir' !!!")
parser.add_argument('--pred_csv', type=str, help="give prediction output csv filename by '--pred_csv' !!!")
args = parser.parse_args()

if __name__ == '__main__':
    image_paths_list = glob(args.image_dir + '*')
    print('Find {} images.'.format(len(image_paths_list)))

    yolo = YOLO()
    yolo.detect_on_set(image_paths_list = image_paths_list, output_csv_name = args.pred_csv, object='person', save_animation=True)
    yolo.close_session()
