# infer
python infer.py -d /PATH/SemanticKITTI/dataset -l logs/2020-5-05-10_37/infer -m logs/2020-5-05-10_37 -s True --gpu 0

## evaluate
python evaluate_iou.py -d /PATH/SemanticKITTI/dataset -p logs/2020-5-05-10_37/infer
