python3 detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source /home/u20/SMOKE/examples/000024.png


python3 train.py  --device 0 --batch-size 1 --data data/car.yaml --img 720 640 \
--cfg cfg/training/yolov7.yaml --weights ./models/yolov7.pt --name v7 --hyp data/hyp.scratch.custom.yaml
sudo python3 train.py  --device 0 --batch-size 1 --data data/yolo.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights /home/u20/yolov7-main/models/yolov7.pt --name yolov7voc --hyp data/hyp.scratch.custom.yaml
