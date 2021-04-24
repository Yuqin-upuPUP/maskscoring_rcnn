# Yuqin LI 20210422
# 选定部分权重文件后，依次对这些权重文件进行预测评估
# 需要手动调整是否启用maskiou进行mrcnn和msrcnn的切换

#!/usr/bin/env python
# coding=UTF-8

import argparse

import os
import numpy as np
import pandas as pd
import json
from skimage.measure import find_contours
from datetime import datetime
import time
import cv2
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from tools.train_net import test
from predictor import COCODemo

# 1.修改后的配置文件
config_file = "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml"

# 2.配置
cfg.merge_from_file(config_file)  # merge配置文件
cfg.merge_from_list(["MODEL.MASK_ON", True])  # 打开mask开关  # .yaml
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])  # or设置为CPU ["MODEL.DEVICE", "cpu"]
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])  # defaults.py

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

weights_folder = 'proposal_weights'  # 待评估预测的模型权重文件夹
in_folder = 'datasets/midea/test/'  # 测试集

tune_root = 'tune'  # 存放所有调参结果的文件目录

cfg.merge_from_list(['DATASETS.TEST', ("coco_midea_test",)])  # 指定评估的数据集为测试集

eval_indicators = ['bbox_AP', 'bbox_AP50', 'bbox_AP75', 'bbox_APs', 'bbox_APm', 'bbox_APl', 'bbox_material_AP',
                   'bbox_material_AP50', 'bbox_material_AP75', 'bbox_material_APs', 'bbox_material_APm',
                   'bbox_material_APl',
                   'segm_AP', 'segm_AP50', 'segm_AP75', 'segm_APs', 'segm_APm', 'segm_APl', 'segm_material_AP',
                   'segm_material_AP50', 'segm_material_AP75', 'segm_material_APs', 'segm_material_APm',
                   'segm_material_APl']


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


def my_evaluate(distributed, visualize=False, save_json=False):
    """
    distributed: 多gpu
    visualize: 进行predict预测，保存渲染mask后的可视化图片
    save_json: 保存predict预测结果annotations

    如果 visualize 和 save_json 都为 False，则只进行coco评价指标的评估
    """

    tune_folder = 'tune-' + timestamp + ('-visualize' if visualize else '') + ('-json' if save_json else '')  # 存放本次调参结果的文件目录，带有本次调参实验的时间戳
    print('-' * 190, '本次调参测试评估实验的结果将存放在：', tune_folder, '-' * 190, sep='\n')

    results_pd = pd.DataFrame(columns=eval_indicators)

    for weight in sorted(os.listdir(weights_folder), key=lambda x: int(x[6:-4])):

        weight_name = os.path.splitext(weight)[0]
        weight_path = os.path.join(weights_folder, weight)
        cfg.merge_from_list(['MODEL.WEIGHT', weight_path])

        print('-' * 190, cfg.MODEL.WEIGHT, '-' * 190, sep='\n')

        coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.5,  # 3.设置置信度
        )

        out_folder = os.path.join(tune_root, tune_folder, weight_name)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # 借鉴训练时对验证集的做法，使用该权重对测试集进行coco评估

        eval_results = test(cfg, coco_demo.model, distributed)  # 会对  cfg.DATASETS.TEST  指定的数据集进行评估
        print('-' * 100, '评估结果')
        eval_value = list(eval_results['bbox'].values()) + list(eval_results['segm'].values())
        # print(eval_value)
        eval_value_series = pd.Series(eval_value, index=eval_indicators, name=weight_name)
        # print(eval_value_series)
        results_pd = results_pd.append(eval_value_series)
        print(results_pd)
        print('-' * 100, '评估结果')

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # 使用该权重对测试集进行预测和json持久化
        if visualize or save_json:
            json_dict = dict()
            images = list()
            annotations = list()
            annotations_midea = list()

            annotation_id = 0

            print('-' * 100, 'predict预测')
            for file_name in tqdm(sorted(os.listdir(in_folder), key=lambda x: int(x[:-4]))):
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

                # method1. 直接得到opencv图片结果
                # predictions = coco_demo.run_on_opencv_image(image)
                # save_path = os.path.join(out_folder, file_name)
                # cv2.imwrite(save_path, predictions)

                # method2. 获取预测结果
                predictions = coco_demo.compute_prediction(image)
                top_predictions = coco_demo.select_top_predictions(predictions)

                # draw
                if visualize:
                    img = coco_demo.overlay_mask(image, top_predictions)
                    img = coco_demo.overlay_boxes(img, top_predictions)
                    img = coco_demo.overlay_class_names(img, top_predictions)
                    save_path = os.path.join(out_folder, file_name)
                    cv2.imwrite(save_path, img)

                # print results
                boxes = top_predictions.bbox.numpy()
                labels = top_predictions.get_field("labels").numpy()  # label = labelList[np.argmax(scores)]
                scores = top_predictions.get_field("scores").numpy()
                material_labels = top_predictions.get_field(
                    "material_labels").numpy()  # label = labelList[np.argmax(scores)]
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
                    annotation['score'] = float(scores[i])
                    annotation['material_id'] = int(material_labels[i])
                    annotation['material_scores'] = float(material_scores[i])
                    annotation['segmentation'] = mask_to_seg(masks[i])
                    annotation['bbox'] = [round(x) for x in boxes[i]]
                    annotation['bbox'][2] = annotation['bbox'][2] - annotation['bbox'][0]
                    annotation['bbox'][3] = annotation['bbox'][3] - annotation['bbox'][1]
                    annotation['area'] = int(areas[i])
                    annotation['iscrowd'] = 0

                    annotations.append(annotation)
                    annotations_midea[-1].append(annotation)

            if save_json:
                json_dict['images'] = images
                json_dict['annotations'] = annotations
                json_dict['categories'] = categories
                json_dict['materials'] = materials

                with open(os.path.join(out_folder, 'data.json'), 'w', encoding='utf-8') as f:
                    json.dump(json_dict, f)

                json_dict['annotations'] = annotations_midea  # 保存完coco形式的json后，再换成midea要求的形式保存一次

                # with open(os.path.join(save_dir, save_name), 'w', encoding='utf-8') as f:
                with open(os.path.join(out_folder, 'data_midea.json'), 'w', encoding='utf-8') as f:
                    json.dump(json_dict, f)
                print('-' * 100, 'predict预测')

    sheet_name = 'msrcnn' if cfg.MODEL.MASKIOU_ON else 'mrcnn'
    excel_path = os.path.join(tune_root, tune_folder, '%s.xlsx' % timestamp)
    with pd.ExcelWriter(excel_path) as writer:
        results_pd.to_excel(writer, sheet_name=sheet_name)


if __name__ == '__main__':
    start = time.time()

    cfg.merge_from_list(['TEST.IMS_PER_BATCH', 1])
    # cfg.merge_from_list(['MODEL.MASKIOU_ON', False])
    print('cfg.TEST.IMS_PER_BATCH:', cfg.TEST.IMS_PER_BATCH)
    print('cfg.MODEL.MASK_ON:', cfg.MODEL.MASK_ON)
    print('cfg.MODEL.MASKIOU_ON:', cfg.MODEL.MASKIOU_ON)

    parser = argparse.ArgumentParser(description='My predition')
    parser.add_argument(
        '--visualize',
        default=False,
        type=bool
    )  # 是否进行可视化图片并保存
    parser.add_argument(
        '--save-json',
        default=False,
        type=bool
    )  # 是否保存预测结果json

    args = parser.parse_args()

    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = num_gpus > 1

    my_evaluate(args.distributed, visualize=args.visualize, save_json=args.save_json)

    end = time.time()
    time_consuming = end - start
    print('耗时：', time_consuming)

