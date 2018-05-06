# Note:
Before any usage, download yolo.h5 file here：https://pan.baidu.com/s/1CxVbByAQS5Oy8noV4GHAiQ key：y7h2, and put it into the directory cfg.
# Usage:
### To make ground truth on a dataset:
  cd ./evaluate/extra/
  ##### RAP ground truth:		
  python RAP_ground_truth.py --input_path <RAP-dataset's ground truth csv file path> \
  hahah
  ##### INRIA ground truth:	
  python INRIA_ground_truth.py --input_dir <INRIA-dataset's annotation directory>
### To detect pedestrians on a dataset: 
  python detect_pedestrian.py --image_dir <dataset's image dir> --pred_csv <prediction output csv filename, e.g. rep_prediction.csv>
### To evaluate mAP:
  python evaluate.py --gt_csv <ground truth csv filename, e.g. rap_ground_truth.csv>  --pred_csv <prediction csv filename, e.g. rap_prediction.csv> --draw_plot <draw result plots or not, e.g. True>
