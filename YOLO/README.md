# usage:
### Make ground truth on RAP-dataset:
  python RAP_ground_truth.py --input_path <RAP-dataset's ground truth csv file path>
### Detect pedestrian on a dataset: 
  python detect_pedestrian.py --image_dir <dataset's image dir>  --gt_csv <ground truth csv filename, e.g. rap_ground_truth.csv>  --result_csv <output result csv filename, e.g. rep_result.csv>
