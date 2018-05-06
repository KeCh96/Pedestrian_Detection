import pandas as pd
from glob import glob
import argparse


gt_path = '../ground_truth/'

# Ground-truth files.csv format:
#     <file_id> <class_name> <bbox> <used>
#     <bbox>: "left top right bottom"

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help="give annotation files directory by '--input_dir' !!!")
args = parser.parse_args()

output = []
anno_paths = glob(args.input_dir + '*')
for path in anno_paths:
    with open(path, 'r') as f:
        content = f.readlines()
    file_id = content[2].split('"')[1].split('/')[-1]
    bbox = content[17].split(':')[-1].replace(') - (', ', ').split(')')[0].split('(')[-1].replace(',', '')
    output.append({"file_id":file_id,  "class_name":'person', "bbox":bbox, "used":False })

output = pd.DataFrame(output)
output.to_csv(gt_path + 'inria_ground_truth.csv', header=True, index=False)
print('Done!')
