import argparse
import random
import numpy as np
import os
from engines.train import Train
from engines.data import DataManager
from engines.configure import Configure
from engines.utils.logger import get_logger
from engines.predict import Predictor


def set_env(configures):
    random.seed(configures.seed)
    np.random.seed(configures.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = configures.CUDA_VISIBLE_DEVICES


def fold_check(configures):
    datasets_fold = 'datasets_fold'
    assert hasattr(configures, datasets_fold), 'item datasets_fold not configured'

    if not os.path.exists(configures.datasets_fold):
        print('datasets fold not found')
        exit(1)

    checkpoints_dir = 'checkpoints_dir'
    if not hasattr(configures, checkpoints_dir):
        os.mkdir('checkpoints')
    else:
        if not os.path.exists(configures.checkpoints_dir):
            print('checkpoints fold not found, creating...')
            os.makedirs(configures.checkpoints_dir)

    vocabs_dir = 'vocabs_dir'
    if not hasattr(configures, vocabs_dir):
        os.mkdir(configures.datasets_fold + '/vocabs')
    else:
        if not os.path.exists(configures.vocabs_dir):
            print('vocabs fold not found, creating...')
            os.makedirs(configures.vocabs_dir)

    log_dir = 'log_dir'
    if not hasattr(configures, log_dir):
        os.mkdir('/logs')
    else:
        if not os.path.exists(configures.log_dir):
            print('log fold not found, creating...')
            os.makedirs(configures.log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with BiLSTM+CRF')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configure(config_file=args.config_file)  # 这里获取了系统config的信息

    fold_check(configs) # 检查config中是否配置了dataset、checkpoint、vocabs、log文件的储存位置
    logger = get_logger(configs.log_dir) # 获取了logger对象，记录日志并输出到文件中
    configs.show_data_summary(logger)  # 将config信息输出到logger中
    set_env(configs) # 配置随机数和device
    mode = configs.mode.lower() # 配置模式
    dataManager = DataManager(configs, logger) # 初始化数据管理器（保存有token和label对向量或索引的映射关系）
    if mode == 'train':
        logger.info('mode: train')
        train = Train(configs, dataManager, logger)  # 完成模型初始化（或已有模型参数的加载）
        train.train()  # 训练时，加载数据集，训练、评估、优化、验证的一系列过程，以及最终模型存储
    elif mode == 'interactive_predict':
        logger.info('mode: predict_one')
        predictor = Predictor(configs, dataManager, logger)  # 完成模型初始化（或已有模型参数的加载）
        predictor.predict_one('warm start')  
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            results = predictor.predict_one(sentence)  # 预测一个句子的输出，先得到每一个字的标签的输出值，再结合原句子得到实体部分、标签部分和位置
            print(results)
    
    elif mode == 'output':
        predictor = Predictor(configs, dataManager, logger)
        predictor.predict_one('warm start')
        import codecs
        input_data = codecs.open("./origindata.txt", "r", 'utf-8')
        output_data = codecs.open("./labeleddata.txt", "w", 'utf-8')
        for line in input_data.readlines():
            output_data.write(line)
            results = predictor.predict_one(line)
            for fea, lab, pos in zip(results[0], results[1], results[2]):
                output_data.write(fea + ':' + lab + ',(' + str(pos[0]) + ',' + str(pos[1]) + ') | ')

            output_data.write('\n\n')
        input_data.close()
        output_data.close()

    elif mode == 'test':
        logger.info('mode: test')
        predictor = Predictor(configs, dataManager, logger)
        predictor.predict_one('warm start')
        predictor.predict_test()
    elif mode == 'save_pb_model':
        logger.info('mode: save_pb_model')
        predictor = Predictor(configs, dataManager, logger)
        predictor.save_pb()
