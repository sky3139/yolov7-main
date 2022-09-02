

import xml.dom.minidom as xml
import abc
import os
import glob
import numpy as np
import data.viz
import ji
import cv2


def gangao():
    predictor = ji.init()
    for i in range(100):
        # original_image = cv2.imread(
        #     '/home/u20/data/training/image_2/%06d.png' % i)   # 读取图片
        original_image=cv2.imread("/home/u20/图片/a.webp")
        result = ji. process_image(predictor, original_image)
        print(result)
        with open('data.json', 'w', encoding='utf-8') as file:
            file.write(result)

gangao()
# #coding : utf-8
# data.viz.ROOT_PATH = "data/1441"
# xmls = glob.glob("%s/*.xml" % data.viz.ROOT_PATH)
# predictor = ji.init()
# for xml_path in xmls:
#     reader = data.viz. XmlTester()
#     im_shape, bbox = reader.load(xml_path)
#     reader.imshow()

#     # original_image = cv2.imread(data.viz.ROOT_PATH+"/"+reader.name)   # 读取图片
#     # result = ji. process_image(predictor, original_image)
#     # print(result)
