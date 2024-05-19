# python3 train.py --data data/widerface.yaml  --cfg cfg/yolov7s-face-tao.yaml --weights ./weights/yolov7s-face.pt --batch-size 16
# python3 train.py --data data/widerface.yaml --cfg models/yoloface-s.yaml --weights 'pretrained models'
# python3 train.py --data data/refineface.yaml  --cfg cfg/yolov7s-face-tao.yaml --weights ./weights/yolov7s-face.pt --batch-size 16

# python3 train.py --data data/refineface.yaml  --cfg cfg/yolov7s-face-tao.yaml --weights /disks/disk1/Workspace/TrainProjs/yolov7-face-main/runs/train/exp10/weights/last.pt --batch-size 16
#


# python3 train.py --data data/dataset_plate.yaml  --cfg cfg/yolov7s-plate.yaml \
#         --hyp    data/hyp.scratch.tiny.yaml   \
#         --batch-size 24  --save_period 5


python3 train.py --data data/dataset_plate.yaml  --cfg cfg/yolov7s-plate.yaml \
        --weights  /disks/disk1/Workspace/TrainProjs/yolo-plate/yolov7-face-main/runs/train/exp19/weights/last.pt \
        --hyp    data/hyp.scratch.tiny.yaml   \
        --batch-size 24  --save_period 5