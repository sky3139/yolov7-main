import json
import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression,  scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

import numpy as np


class MyMode():
    def get_name(self, idx):
        # Car Van Truck Bus Pedestrian Cyclist Motorcyclist Barrow  Tricyclist.
        self.names[int(idx)]
        # if idx in self.ndict:  # .has_key(idx):
        #     return self.ndict[idx]
        # else:
        #     return "no"

    def __init__(self):

        self.ndict = {"2": "car", "7": "truck", "0": "pedestrian",
                      "3": "motorcycle", "5": "bus", "1": "Cyclist"}
        self.ndict["7"]="truck"
        fconf = cv2.FileStorage("config.yml", cv2.FileStorage_READ)
        weights = fconf.getNode('weights').string()
        self.conf_thres = fconf.getNode('conf_thres').real()
        self.iou_thres = fconf.getNode('iou_thres').real()
        self.classes = None
        self.agnostic_nms = True
        fconf.release()
        self.imgsz = 640
        device = select_device('0')
        self.device = device
        # Load model
        self.model = attempt_load(
            weights, map_location=device)  # load FP32 self.model
        self.stride = int(self.model.stride.max())  # self.model stride
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check img_size
        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        # self.names_
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self. imgsz).to(
                device).type_as(next(self.model.parameters())))  # run once

    def __call__(self, im0s):
        assert im0s is not None, 'Image Not Found '
        # Padded resize
        img = letterbox(im0s, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            dets = self.model(img, augment=True)[0]
        pred = non_max_suppression(
            dets, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        # Process detections
        det = pred[0]
        s = ''
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0s.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # add to string
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

            # Write results
            ans = []
            for x1, y1, x2, y2, score, cls in det.cpu().numpy():
                _name = self.get_name(str(int(cls)))
                jieduan = 0
                if score < 0.6:
                    zhedang = 2
                elif score < 0.85:
                    jieduan = 1
                    zhedang = 1
                else:
                    zhedang = 0
                at = [_name, jieduan, zhedang, 0.5, x1,
                      y1, x2, y2, 1.0, 2.0, 3., 4., 5., 2., 1.0]
                if _name == "no":
                    print(_name, self.names[int(cls)], cls)
                else:
                    ans.append([str(v) for v in at])
            # print("a",det.cpu().numpy())

            for *xyxy, conf, cls in reversed(det):
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label,
                             color=self.colors[int(cls)], line_thickness=1)
        # Stream results
        cv2.imshow("aaa", im0s)
        cv2.waitKey(0)  # 1 millisecond
        return ans
        # print(f'Done. ({time.time() - t0:.3f}s)')
        # print(dets)


def init():       # 模型初始化
    model = MyMode()  # "您的深度学习模型"  ###开发者需要自行设计深度学习模型
    return model


def process_image(net, input_image, args=None):
    dets = net(input_image)
    result = {"model_data": {"objects": dets}}
    return json.dumps(result)


if __name__ == '__main__':

    predictor = init()
    original_image = cv2.imread('/home/u20/SMOKE/examples/000024.png')   # 读取图片
    result = process_image(predictor, original_image)
    # print(result)
    with open('data.json', 'w', encoding='utf-8') as file:
        file.write(result)
