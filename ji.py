import json
import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression,  scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from PIL import Image
import numpy as np


class MyMode():
    def __init__(self):
        pass

    def __call__(self, im0s):
        ans = {"object_detect": [[1, 2, 3, 4, "open_bed_heavy_truck", 0.3]],
               "segment": np.array((3, 3),dtype=np.uint8)}
        return ans
        # print(f'Done. ({time.time() - t0:.3f}s)')
        # print(dets)


def init():       # 模型初始化
    model = MyMode()  # "您的深度学习模型"  ###开发者需要自行设计深度学习模型
    return model


def process_image(net, input_image, args=None):
    results = net(input_image)

    detect_objs = []
    for k, det in enumerate(results['object_detect']):
        x, y, width, height, name, score = det
        obj = {
            'name': name,
            'xmin': x,
            'ymin': y,
            'width': width,
            'height': height, 'confidence': float(score)
        }
        if name == 'open_bed_heavy_truck':
            '''
            开放式大型货车,需要识别车辆颜色，还有检测车牌的4个角点，开发者需要自行设计这些模型
            '''
            obj['color'] = 'white'
            obj['plate'] = [{'name': 'back_plate', 'points': [608, 610, 943, 713, 934, 773, 607, 664], 'ocr': 'xxxxxxx',
                             'confidence': 0.8},
                            {'name': 'size_plate', 'points': [594, 596, 929, 700, 930, 761, 594, 650], 'ocr': 'xxxxxxx',
                             'confidence': 0.8}
                            ]
            '''
            开放式大型货车身上，可能有多个车牌的，因此'plate'的值是一个列表，里面可以存放多个车牌
            车牌的ocr信息不在评测范围里，开发者可以输出这个信息，也可以不输出
            '''
        detect_objs.append(obj)

    mask = results['segment']
    args = json.loads(args)
    mask_output_path = args['mask_output_path']
    pred_mask_per_frame = Image.fromarray(mask)
    pred_mask_per_frame.save(mask_output_path)

    pred = {'model_data': {"objects": detect_objs, "mask": mask_output_path}}
    return json.dumps(pred)


if __name__ == '__main__':

    predictor = init()
    original_image = cv2.imread('/home/u20/SMOKE/examples/000024.png')   # 读取图片
    args = {"mask_output_path": "mask.png"}
    
    result = process_image(predictor, original_image, json.dumps(args))
    # print(result)
    with open('data.json', 'w', encoding='utf-8') as file:
        file.write(result)
