# infer
python infer.py -d /PATH/SemanticKITTI/dataset -l logs/2021-1-16-09:00/infer -m logs/2021-1-16-09:00 -s True --gpu 0

## evaluate
python evaluate_iou.py -d /PATH/SemanticKITTI/dataset -p logs/2021-1-16-09:00/infer
