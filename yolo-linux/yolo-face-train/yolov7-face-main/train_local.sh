# python3 train.py --data data/widerface.yaml  --cfg cfg/yolov7s-face-tao.yaml --weights ./weights/yolov7s-face.pt --batch-size 16
# python3 train.py --data data/widerface.yaml --cfg models/yoloface-s.yaml --weights 'pretrained models'
# python3 train.py --data data/refineface.yaml  --cfg cfg/yolov7s-face-tao.yaml --weights ./weights/yolov7s-face.pt --batch-size 16

# python3 train.py --data data/refineface.yaml  --cfg cfg/yolov7s-face-tao.yaml --weights /disks/disk1/Workspace/TrainProjs/yolov7-face-main/runs/train/exp10/weights/last.pt --batch-size 16
#

python3 train.py --data data/widerface.yaml  --cfg cfg/yolov7s-face.yaml \
        --hyp    data/hyp.scratch.tiny.yaml   \
        --batch-size 16  --save_period 5