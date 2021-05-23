import argparse

import os
from datetime import datetime
import time

import torch
from maskrcnn_benchmark.config import cfg

from my_test_net import my_evaluate

# 配置------------------------------------------------------------------------------------------------------------------

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

MSRCNN = True

# 训练配置---------------------------------------------------------------------------------------------------------------
# MRCNN_CONFIG_FILE = 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'  # mrcnn配置文件
# MSRCNN_CONFIG_FILE = 'configs/e2e_ms_rcnn_R_50_FPN_1x.yaml'  # msrcnn配置文件
# CONFIG_FILE = MSRCNN_CONFIG_FILE if MSRCNN else MRCNN_CONFIG_FILE

CONFIG_FILE = 'configs/e2e_ms_rcnn_R_50_FPN_1x.yaml'


LOCAL_RANK = 0  #

# 测试配置---------------------------------------------------------------------------------------------------------------
TEST_DATASETS = ("coco_midea_test",)  # 测试阶段评估时的测试集

IN_FOLDER = 'datasets/midea/test/'


TUNE_ROOT = 'predict'  # 整个项目评估测试集的结果存放路径
TUNE_FOLDER = os.path.join(TUNE_ROOT, timestamp)  # 本次模型参数配置条件下评估测试集的结果存放路径

# EVAL_WEIGHTS_ROOT = 'eval_weights'
# EVAL_WEIGHTS_PATH = os.path.join(EVAL_WEIGHTS_ROOT, NAME_STR_WITH_PARAMETERS)  # 准备用来评估的权重文件夹路径，需要先复制保存到这里，再进行评估
EXCEL_PATH = os.path.join(TUNE_FOLDER, '%s.xlsx' % timestamp)  # 测试集评估结果保存excel文件路径
SHEET_NAME = 'msrcnn' if MSRCNN else 'mrcnn'  # 保存到excel中的sheet的名字

# 配置------------------------------------------------------------------------------------------------------------------


def main():

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1


if __name__ == '__main__':
    start = time.time()

    cfg.merge_from_list(['TEST.IMS_PER_BATCH', 1])
    # cfg.merge_from_list(['MODEL.MASKIOU_ON', False])
    print('cfg.TEST.IMS_PER_BATCH:', cfg.TEST.IMS_PER_BATCH)
    print('cfg.MODEL.MASK_ON:', cfg.MODEL.MASK_ON)
    print('cfg.MODEL.MASKIOU_ON:', cfg.MODEL.MASKIOU_ON)

    parser = argparse.ArgumentParser(description='My predition')
    parser.add_argument(
        '--weights-path',
        default=False,
        type=bool
    )  # 是否进行可视化图片并保存
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

    if args.distributed:
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(CONFIG_FILE)
    if not MSRCNN:
        cfg.merge_from_list(["MODEL.MASKIOU_ON", False])
    cfg.freeze()

    # 评估测试集阶段------------------------------------------------------------------------------------------------------
    cfg.merge_from_list(['DATASETS.TEST', TEST_DATASETS])  # 指定评估的数据集为测试集

    # 输出待评估的权重
    # print('\n'.join(os.listdir(EVAL_WEIGHTS_PATH)))
    print('-' * 190, '***** 待评估权重 *****', '-' * 190, sep='\n')
    PROPOSAL_EVAL_WEIGHT = list()
    for path in args.weights_path:
        if path.endswith('.pth'):
            PROPOSAL_EVAL_WEIGHT.append(path)
    print('\n'.join(PROPOSAL_EVAL_WEIGHT))

    print('-' * 190, '***** 评估测试集 *****', '-' * 190, sep='\n')
    my_evaluate(args.distributed,
                cfg,
                IN_FOLDER,
                args.weights_path,
                PROPOSAL_EVAL_WEIGHT,
                TUNE_FOLDER,
                EXCEL_PATH,
                SHEET_NAME,
                visualize=args.visualize,
                save_json=args.save_json
                )

    end = time.time()
    time_consuming = end - start
    print('耗时：', time_consuming)

