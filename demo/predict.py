#!/usr/bin/env python
# coding=UTF-8

import os
import numpy as np
import json
import cv2
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

from skimage.measure import find_contours

from maskrcnn_benchmark.utils import cv2_util

# 1.修改后的配置文件
config_file = "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml"

# 2.配置
cfg.merge_from_file(config_file)  # merge配置文件
cfg.merge_from_list(["MODEL.MASK_ON", True])  # 打开mask开关
# cfg.merge_from_list(["MODEL.DEVICE", "cuda"])  # or设置为CPU ["MODEL.DEVICE", "cpu"]
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

print(cfg.MODEL.WEIGHT)

def mask_to_seg(mask):
    mask = mask[0]
    seg = []
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        seg.append(verts.tolist())
    return seg

# def mask_to_seg(mask):
#     seg = []
#     thresh = mask[0, :, :, None].astype(np.uint8)
#     contours, hierarchy = cv2_util.findContours(
#         thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
#     )
#     for verts in contours:
#         seg.extend(verts.tolist())
# 
#     seg = list(map(int, np.array(seg).flatten()))
#     return [seg]

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,  # 3.设置置信度
)

categories = [
    {"supercategory": "tableware", "id": 1, "name": "bowl"},
    {"supercategory": "tableware", "id": 2, "name": "cup"},
    {"supercategory": "tableware", "id": 3, "name": "plate"},
    {"supercategory": "tableware", "id": 4, "name": "red-wine"},
    {"supercategory": "tableware", "id": 5, "name": "milk pan"},
    {"supercategory": "tableware", "id": 6, "name": "frying pan"},
    {"supercategory": "tableware", "id": 7, "name": "stockpot"},
    {"supercategory": "tableware", "id": 8, "name": "Wok"},
    {"supercategory": "tableware", "id": 9, "name": "enamel"}
]

materials = [
    {"supercategory": "material", "id": 1, "name": "pottery and porcelain"},
    {"supercategory": "material", "id": 2, "name": "glass"},
    {"supercategory": "material", "id": 3, "name": "stainless steel"},
    {"supercategory": "material", "id": 4, "name": "plastics"},
    {"supercategory": "material", "id": 5, "name": "woody"},
    {"supercategory": "material", "id": 6, "name": "other"}
]

if __name__ == '__main__':

    in_folder = 'datasets/midea/test/'
    out_folder = 'predict_results/50000_1'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    json_dict = dict()
    images = list()
    annotations = list()
    annotations_midea = list()

    annotation_id = 0

    print('-'*190, sorted(os.listdir(in_folder), key=lambda x:int(x[:-4])), '-'*190, sep='\n')
    for file_name in sorted(os.listdir(in_folder), key=lambda x:int(x[:-4])):
        if not file_name.endswith(('jpg', 'png', 'bmp')):
            continue

        # load file
        img_path = os.path.join(in_folder, file_name)
        image = cv2.imread(img_path)

        image_dict = dict()
        image_dict['id'] = len(images) + 1  # 从1开始
        image_dict['height'], image_dict['width'], _ = image.shape  # (960, 1280, 3)
        image_dict['file_name'] = file_name

        images.append(image_dict)

        print('-'*50, img_path)

        # method1. 直接得到opencv图片结果
        #predictions = coco_demo.run_on_opencv_image(image)
        #save_path = os.path.join(out_folder, file_name)
        #cv2.imwrite(save_path, predictions)

        # method2. 获取预测结果
        predictions = coco_demo.compute_prediction(image)
        top_predictions = coco_demo.select_top_predictions(predictions)

        # draw
        img = coco_demo.overlay_mask(image, top_predictions)
        img = coco_demo.overlay_boxes(img, top_predictions)
        img = coco_demo.overlay_class_names(img, top_predictions)
        save_path = os.path.join(out_folder, file_name)
        cv2.imwrite(save_path, img)

        # print results
        boxes = top_predictions.bbox.numpy()
        labels = top_predictions.get_field("labels").numpy()  # label = labelList[np.argmax(scores)]
        scores = top_predictions.get_field("scores").numpy()
        material_labels = top_predictions.get_field("material_labels").numpy()  # label = labelList[np.argmax(scores)]
        material_scores = top_predictions.get_field("material_scores").numpy()
        masks = top_predictions.get_field("mask").numpy()
        areas = top_predictions.area()


        annotations_midea.append(list())
        for i in range(len(boxes)):
            annotation = dict()
            annotation['id'] = annotation_id  # 整个测试集上计数
            annotation_id += 1
            annotation['objectId'] = i + 1  # 每张图片独立计数
            annotation['image_id'] = image_dict['id']
            annotation['category_id'] = int(labels[i])
            annotation['material_id'] = int(material_labels[i])
            annotation['segmentation'] = mask_to_seg(masks[i])
            annotation['bbox'] = [round(x) for x in boxes[i]]
            annotation['bbox'][2] = annotation['bbox'][2] - annotation['bbox'][0]
            annotation['bbox'][3] = annotation['bbox'][3] - annotation['bbox'][1]
            annotation['area'] = int(areas[i])
            annotation['iscrowd'] = 0

            annotations.append(annotation)
            annotations_midea[-1].append(annotation)


    json_dict['images'] = images
    json_dict['annotations'] = annotations
    json_dict['categories'] = categories
    json_dict['materials'] = materials

    # with open(os.path.join(save_dir, save_name), 'w', encoding='utf-8') as f:
    with open(os.path.join(out_folder, 'data_0.json'), 'w', encoding='utf-8') as f:
        json.dump(json_dict, f)


    json_dict['annotations'] = annotations_midea  # 保存完coco形式的json后，再换成midea要求的形式保存一次

    # with open(os.path.join(save_dir, save_name), 'w', encoding='utf-8') as f:
    with open(os.path.join(out_folder, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(json_dict, f)
