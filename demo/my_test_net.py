#!/usr/bin/env python
# coding=UTF-8

import os
import numpy as np
import pandas as pd
import json
from skimage.measure import find_contours
import cv2
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from tools.train_net import test
from predictor import COCODemo


CATEGORIES = [
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

MATERIALS = [
    {"supercategory": "material", "id": 1, "name": "pottery and porcelain"},
    {"supercategory": "material", "id": 2, "name": "glass"},
    {"supercategory": "material", "id": 3, "name": "stainless steel"},
    {"supercategory": "material", "id": 4, "name": "plastics"},
    {"supercategory": "material", "id": 5, "name": "woody"},
    {"supercategory": "material", "id": 6, "name": "other"}
]

EVAL_INDICATORS = ['bbox_AP', 'bbox_AP50', 'bbox_AP75', 'bbox_APs', 'bbox_APm', 'bbox_APl', 'bbox_material_AP',
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


# 选定部分权重文件后，依次对这些权重文件进行预测评估
def my_evaluate(distributed, in_folder, weights_output_dir, proposal_eval_weight, tune_folder, excel_path, sheet_name, visualize=False, save_json=False):
    """
    distributed: 多gpu
    in_folder: 'datasets/midea/test/' 测试集
    weights_output_dir: 训练过程中保存的权重文件夹路径,
    proposal_eval_weight: 候选的权重文件名称
    tune_folder: 本次模型参数配置条件下评估测试集的结果存放路径,
    excel_path: 本次模型参数配置条件下评估测试集的结果记录excel路径
    sheet_name: excel里sheet的名称
    visualize: 进行predict预测，保存渲染mask后的可视化图片
    save_json: 保存predict预测结果annotations

    如果 visualize 和 save_json 都为 False，则只进行coco评价指标的评估
    """

    print('-' * 190, '本次调参测试评估实验的结果将存放在：', tune_folder, '-' * 190, sep='\n')

    results_pd = pd.DataFrame(columns=EVAL_INDICATORS)

    for weight in sorted(filter(lambda filename: filename.endswith('.pth'), os.listdir(weights_output_dir)), key=lambda x: int(x[6:-4])):

        if not (weight in proposal_eval_weight):
            continue

        weight_name = os.path.splitext(weight)[0]
        weight_path = os.path.join(weights_output_dir, weight)
        cfg.merge_from_list(['MODEL.WEIGHT', weight_path])  # 设置要评估的权重文件

        print('-' * 95, '正在进行评估的权重文件', cfg.MODEL.WEIGHT, '-' * 95, sep='\n')

        coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.5,  # 设置置信度
            my_test=True
        )

        out_folder = os.path.join(tune_folder, weight_name)  # 预测得到的可视化结果存储路径
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # 借鉴训练时对验证集的做法，使用该权重对测试集进行coco评估

        eval_results = test(cfg, coco_demo.model, distributed, weight_name=weight_name)  # 会对  cfg.DATASETS.TEST  指定的数据集进行评估
        print('-' * 48, '评估结果')
        eval_value = list(eval_results['bbox'].values()) + list(eval_results['segm'].values())
        # print(eval_value)
        eval_value_series = pd.Series(eval_value, index=EVAL_INDICATORS, name=weight_name)
        # print(eval_value_series)
        results_pd = results_pd.append(eval_value_series)
        print(results_pd)
        print('-' * 48, '评估结果')

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

            print('-' * 48, 'predict预测')
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
                json_dict['categories'] = CATEGORIES
                json_dict['materials'] = MATERIALS

                with open(os.path.join(out_folder, 'data.json'), 'w', encoding='utf-8') as f:
                    json.dump(json_dict, f)

                json_dict['annotations'] = annotations_midea  # 保存完coco形式的json后，再换成midea要求的形式保存一次

                # with open(os.path.join(save_dir, save_name), 'w', encoding='utf-8') as f:
                with open(os.path.join(out_folder, 'data_midea.json'), 'w', encoding='utf-8') as f:
                    json.dump(json_dict, f)
            print('-' * 48, 'predict预测')

    with pd.ExcelWriter(excel_path) as writer:
        results_pd.to_excel(writer, sheet_name=sheet_name)

