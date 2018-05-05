import os
import argparse
from mAP import eval_mAP



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
result_dir = './evaluate/result_evaluate/'
pred_dir = './evaluate/predicted/'
gt_dir = './evaluate/ground_truth/'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--pred_csv', type=str, help="give prediction file name by '--pred_csv' !!!")
parser.add_argument('--gt_csv', type=str, help="give ground truth file name by '--gt_csv' !!!")
parser.add_argument('--draw_plot', type=bool, help="draw result plots or not")
args = parser.parse_args()



if __name__ == '__main__':

    eval_mAP(
        pred_file_path= pred_dir + args.pred_csv,
        gt_file_path= gt_dir + args.gt_csv,
        output_dir= result_dir,
        draw_plot= args.draw_plot
    )