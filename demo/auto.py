import os
from datetime import datetime

import torch
from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger

from tools.train_net import train, test
from my_test_net import my_evaluate

# 配置------------------------------------------------------------------------------------------------------------------

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

MSRCNN = True

# 训练配置---------------------------------------------------------------------------------------------------------------
MRCNN_CONFIG_FILE = 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'  # mrcnn配置文件
MSRCNN_CONFIG_FILE = 'configs/e2e_ms_rcnn_R_50_FPN_1x.yaml'  # msrcnn配置文件
CONFIG_FILE = MSRCNN_CONFIG_FILE if MSRCNN else MRCNN_CONFIG_FILE


LOCAL_RANK = 0  #
SKIP_TEST = False  # 跳过评估验证集
USE_TENSORBOARD = True


OPTS_DICT = {
    'SOLVER.IMS_PER_BATCH': '2',
    'SOLVER.BASE_LR': '0.0025',
    'SOLVER.MAX_ITER': '600',
    'SOLVER.STEPS': (400, 500),
    'SOLVER.CHECKPOINT_PERIOD': '10',
    'TEST.IMS_PER_BATCH': '1',  # 只能为1
}

# 训练后存放的权重文件
NAME_STR_WITH_PARAMETERS = '%s_BS_%s_LR_%s_MI_%d_%d_%s' % \
                           (
                                timestamp,
                                OPTS_DICT['SOLVER.IMS_PER_BATCH'],
                                OPTS_DICT['SOLVER.BASE_LR'],
                                OPTS_DICT['SOLVER.STEPS'][0],
                                OPTS_DICT['SOLVER.STEPS'][1],
                                OPTS_DICT['SOLVER.MAX_ITER'],
                            )

OPTS_DICT.update({
    'OUTPUT_DIR': 'models/%s' % NAME_STR_WITH_PARAMETERS
})  # 训练后存放的权重文件

OPTS_DICT.update({
    'SOLVER.STEPS': '(%d, %d)' % OPTS_DICT['SOLVER.STEPS']
})

# 模仿 args.opts 组合成list
# type(args.opts): <class 'list'>
# args.opts: [
#   'SOLVER.IMS_PER_BATCH', '2',
#   'SOLVER.BASE_LR', '0.0025',
#   'SOLVER.MAX_ITER', '600',
#   'SOLVER.STEPS', '(400, 500)',
#   'SOLVER.CHECKPOINT_PERIOD', '10',
#   'TEST.IMS_PER_BATCH', '1'
# ]
opts = []
for k, v in OPTS_DICT.items():
    opts.append(k)
    opts.append(v)

# 测试配置---------------------------------------------------------------------------------------------------------------
# 打算评估的权重列表  权重名: model_0125000.pth  这里只提供数字即可
PROPOSAL_EVAL_WEIGHT_NAMES = [
    '0050000',
    '0075000',
    '0100000',
    '0125000'
]

TEST_VISUALIZE = False  # 预测时是否进行可视化并保存为图片
TEST_SAVE_JSON = False  # 是否将预测得到的annotation保存为json

IN_FOLDER = 'datasets/midea/test/'


TUNE_ROOT = 'tune'  # 整个项目评估测试集的结果存放路径
TUNE_FOLDER = os.path.join(TUNE_ROOT, NAME_STR_WITH_PARAMETERS)  # 本次模型参数配置条件下评估测试集的结果存放路径

EVAL_WEIGHTS_ROOT = 'eval_weights'
EVAL_WEIGHTS_PATH = os.path.join(EVAL_WEIGHTS_ROOT, NAME_STR_WITH_PARAMETERS)  # 准备用来评估的权重文件夹路径，需要先复制保存到这里，再进行评估
EXCEL_PATH = os.path.join(TUNE_FOLDER, '%s.xlsx' % NAME_STR_WITH_PARAMETERS)  # 测试集评估结果保存excel文件路径
SHEET_NAME = 'msrcnn' if MSRCNN else 'mrcnn'  # 保存到excel中的sheet的名字

# 配置------------------------------------------------------------------------------------------------------------------


def main():

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(CONFIG_FILE)
    cfg.merge_from_list(opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(CONFIG_FILE))
    with open(CONFIG_FILE, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 训练阶段-----------------------------------------------------------------------------------------------------------
    # model = train(cfg, args.local_rank, distributed)
    print('-' * 190, '***** 训练阶段 *****', '-' * 190, sep='\n')
    model = train(
        cfg=cfg,
        local_rank=LOCAL_RANK,
        distributed=distributed,
        use_tensorboard=USE_TENSORBOARD,
        name_str_with_parameters=NAME_STR_WITH_PARAMETERS
    )

    # 评估验证集阶段------------------------------------------------------------------------------------------------------
    if not SKIP_TEST:
        print('-' * 190, '***** 评估验证集 *****', '-' * 190, sep='\n')
        logger.info("Evaluate on the val dataset")
        test(cfg, model, distributed)

    # 评估测试集阶段------------------------------------------------------------------------------------------------------
    print('-' * 190, '***** 转移待评估权重文件 *****', '-' * 190, sep='\n')
    if not os.path.exists(EVAL_WEIGHTS_PATH):
        os.makedirs(EVAL_WEIGHTS_PATH)
    for weight_name in PROPOSAL_EVAL_WEIGHT_NAMES:
        weight_path = 'model_%s.pth' % weight_name  # model_0125000.pth
        origin_weight_path = os.path.join(OPTS_DICT['OUTPUT_DIR'], weight_path)
        cp_command = 'cp %s %s' % (origin_weight_path, EVAL_WEIGHTS_PATH)
        os.system(cp_command)

    # 输出待评估的权重
    print('\n'.join(os.listdir(EVAL_WEIGHTS_PATH)))

    print('-' * 190, '***** 评估测试集 *****', '-' * 190, sep='\n')
    my_evaluate(distributed,
                IN_FOLDER,
                EVAL_WEIGHTS_PATH,
                TUNE_FOLDER,
                EXCEL_PATH,
                SHEET_NAME,
                visualize=TEST_VISUALIZE,
                save_json=TEST_SAVE_JSON
                )


if __name__ == "__main__":
    main()
