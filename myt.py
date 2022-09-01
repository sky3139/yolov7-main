

import ji
import cv2
predictor = ji.init()


for i in range(100):
    original_image = cv2.imread('/home/u20/data/training/image_2/%06d.png'%i)   # 读取图片
    result =ji. process_image(predictor, original_image)
    print(result)
    with open('data.json', 'w', encoding='utf-8') as file:
        file.write(result)
    