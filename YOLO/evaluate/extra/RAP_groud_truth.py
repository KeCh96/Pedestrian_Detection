import argparse
import pandas as pd

gt_path = '../ground_truth/'


#
# Ground-truth files.csv format:
#     <file_id> <class_name> <bbox> <used>
#     <bbox>: "left top right bottom"
# Predicted objects files.csv format:
#     <file_id> <class_name> <bbox> <confidence>
#


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help="give raw ground truth path by '--input_path' !!!")
args = parser.parse_args()


data = pd.read_csv(args.input_path)
output = []
for _, row in data.iterrows():
    file_name = row['filename'].split('.')[0]
    left = row['x']
    right = row['x'] + row['width']
    top = row['y']
    bottom = row['y'] + row['height']
    bbox = str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
    output.append({"file_id":file_name,  "class_name":'person', "bbox":bbox, "used":False })

output = pd.DataFrame(output)
output.to_csv(gt_path + 'rap_ground_truth.csv', header=True, index=False)
print('Done!')