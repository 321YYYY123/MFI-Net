"""
Author: Benny
Date: Nov 2019
"""

from dataset import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf
from torchvision.utils import save_image
import datetime as dt

from pointnet_util import voxelize_point_cloud



MODE="Train"  # 选择模式 "Train" 或 "Test"
preModepath = r"log\cls\Hengshuang\MiniShift1024_02242354\last.pth" #r"log\cls\PTM\Air04250118\best.pth" # 预训练模型路径，如果是测试模式需要指定预训练模型路径

SelectDataSet = 5  # 选择数据集的索引，0表示Airplane，1表示Car，2表示Chair，3表示ModelNet40
saveTh=False # 是否保存特定实例精度的模型
LoadLr= 1  # 是否加载学习率
batch_size = 10




#auto select dataset and set parameters
PtLS=['Data\FG3d_Airplane1024',"Data\FG3d_Car1024","Data\FG3d_Chair1024","Data\modelnet40_normal_resampled","Data\gap_smallguandao","Data\MiniShift1024"]
ClassNumLst=[13, 20, 33, 40,5,12]  # 每个数据集对应的类别数
Data_Path=PtLS[SelectDataSet]  # 默认数据集路径
NumClass= ClassNumLst[SelectDataSet]  # 类别数

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 关键设置
    torch.backends.cudnn.benchmark = False  # 禁用性能优化
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()

def test(model, loader, num_class):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target, _ = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(
                target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


@hydra.main(version_base=None, config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    args.input_dim = 6 if args.normal else 3
    args.num_class = NumClass
    args.batch_size = batch_size
    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path(Data_Path)
    DataName = Path(DATA_PATH).name
    dl="," if DataName == 'modelnet40_normal_resampled' else " "  # 分割方式
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH,
                                       npoint=args.num_point,
                                       split='train',
                                       normal_channel=args.normal,
                                       dl=dl)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH,
                                      npoint=args.num_point,
                                      split='test',
                                      normal_channel=args.normal,
                                      dl=dl)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
    '''MODEL LOADING'''

    shutil.copy(
        hydra.utils.to_absolute_path('models/{}/model.py'.format(
            args.model.name)), '.')

    classifier = getattr(
        importlib.import_module('models.{}.model'.format(args.model.name)),
        'PointTransformerCls')(
            args).cuda()  #根据配置文件中选择的model导入相应的分类网络结构 并实例化传入args
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(args)

    #拼接时间 创建路径
    StyleTime = dt.datetime.now().strftime("%m%d%H%M")
   
    ModelSavePath = f"log\cls\{args.model.name}\{DataName}_{StyleTime}/"
    os.makedirs(ModelSavePath, exist_ok=True)  #创建文件夹
    checkpoint = None
        # 创建或清空loss记录文件
    with open(ModelSavePath+'loss_lr_aa_oa_log.txt', 'w') as f:
        f.write('epoch,loss,lr,aa,oa\n')  # 写入表头

    
    try:

        #checkpoint = torch.load(ModelSavePath + "best.pth")
        checkpoint = torch.load(preModepath)  #想继续训练把模型放这个路径下
        start_epoch = checkpoint[
            'epoch']  #加载checkpoint中保存的训练到的epoch作为继续训练的起始epoch
        classifier.load_state_dict(
            checkpoint['model_state_dict'])  #载入模型权重给分类模型
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    #根据配置文件选择 优化器
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,  #学习率
            betas=(0.9, 0.999),  #beta1=0.9, beta2=0.999 作用是计算梯度平方的指数加权平均数
            eps=1e-08,  #防止分母为0
            weight_decay=args.weight_decay)  #weight_decay 权重衰减
    else:

        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9)  #momentum 动量 0.9

    if checkpoint != None and LoadLr:
        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])  # 载入优化器的状态字典，包括学习率
    #设置可变衰减的学习率


# 统计网络参数量
    total_params = sum(p.numel() for p in classifier.parameters())
    logger.info('Total number of parameters: %.6fM' % (total_params / 1e6))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=50,
                                                gamma=0.3)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []
    lossEpoch = 0
    lossNum = 0
    
    '''TRANING'''
    logger.info('Start training...')
    
    for epoch in range(start_epoch, start_epoch + args.epoch):
        logger.info('Epoch %d (%d/%s):' %
                    (global_epoch + 1, epoch + 1, start_epoch + args.epoch))
        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info('Current Learning Rate: %.10f' % current_lr)

        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0),
                                   total=len(trainDataLoader),
                                   smoothing=0.9):
            points, target, _ = data

            points = points.data.numpy()
            points = provider.random_point_dropout(points)  #随机点云丢弃
            points[:, :,
                   0:3] = provider.random_scale_point_cloud(points[:, :,
                                                                   0:3])  #随机缩放
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :,
                                                                  0:3])  #随机平移

            points = torch.Tensor(points)

            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            lossEpoch += loss.item()
            lossNum += 1

        scheduler.step()
        lossEpoch = lossEpoch / lossNum if lossNum > 0 else 0
        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader,
                                           args.num_class)
  
            #最好的模型保存
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                logger.info('Save Best model...')
                savepath = ModelSavePath + f'best.pth'
                logger.info('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'config': args,
                    'datapath': DATA_PATH,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                
    

            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' %
                        (instance_acc, class_acc))  #OA和AA 
         
            with open(ModelSavePath+'loss_lr_aa_oa_log.txt', 'a') as f:
                f.write(f'{epoch},{lossEpoch:.6f},{current_lr:.10f},{class_acc:.6f},{instance_acc:.6f}\n')

                
            logger.info(
                'Best Instance Accuracy: %f, Class Accuracy: %f, Epoch: %d' %
                (best_instance_acc, best_class_acc, best_epoch))

            logger.info('Save Last model...')
            savepath = ModelSavePath + 'last.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch + 1,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'config': args,
                'datapath': DATA_PATH,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    if MODE=='Train':
        print("Training Mode")
        main()
    else:
        print("Testing Mode")
        from test_cls import TestModelMain
        TestModelMain(preModepath)
