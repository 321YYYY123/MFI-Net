import importlib
import logging
from operator import truediv
import os
import random
import hydra
import torch
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
from dataset import ModelNetDataLoader
import yaml
import matplotlib.pyplot as plt
from pointnet_util import plot_pcd
from omegaconf import OmegaConf
import seaborn as sns
from thop import profile, clever_format

ModelPATH=r'log\cls\Menghao\gap_smallguandao_07102359\best.pth'



def visualize_misclassified_point_cloud(points,
                                        true_label,
                                        pred_label,
                                        class_names,
                                        save_path=None):
    """
    可视化错分类的点云
    Args:
        points: 点云数据 (N, 3)
        true_label: 真实标签
        pred_label: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径，如果为None则显示图像
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    plot_pcd(ax, points)

    # 设置标题
    title = f'True: {class_names[true_label]}, Pred: {class_names[pred_label]}'
    ax.set_title(title)

    # 设置视角
    ax.view_init(elev=30, azim=45)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def SaveTestReport(y_pred, y_test, file_name, classname=None, flops=None, params=None, avg_inference_time=None):
    oa = accuracy_score(y_test, y_pred)
    # 得到混淆矩阵中的准确度
    confusion = confusion_matrix(y_test, y_pred)
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    aa = np.mean(each_acc)
    ####
    # kappa = cohen_kappa_score(y_test, y_pred)
    ###
    if classname is None:
        classification = classification_report(
            y_test, 
            y_pred, 
            digits=4,
        )
    else:
        classification = classification_report(
            y_test, 
            y_pred, 
            digits=4,
            target_names=classname,
        )
    # 写入文件中#
    file_namet = file_name + '_classification_report.txt'
    tqdm.write("\nclassification_report save in " + file_namet + '\n')
    with open(file_namet, 'w') as x_file:
        x_file.write('{} OA(%)'.format(oa * 100))
        x_file.write('\n')
        x_file.write('{} AA (%)'.format(aa * 100))
        x_file.write('\n')
        # x_file.write('{} Kappa(%)'.format(kappa * 100))
        # x_file.write('\n')
        
        # 添加模型性能指标
        if flops and params:
            x_file.write(f"Model FLOPs: {flops}\n")
            x_file.write(f"Model Parameters: {params}\n")
        if avg_inference_time:
            x_file.write(f"Average Inference Time per Sample: {avg_inference_time:.4f} ms\n")
            
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('\nConfusion Matrix:\n')

        # 找到每列的最大宽度
        col_widths = [
            max(len(str(confusion[i, j])) for i in range(confusion.shape[0]))
            for j in range(confusion.shape[1])
        ]

        # 保存混淆矩阵，确保列左对齐
        for row in confusion:
            row_str = '  '.join(f'{row[j]:<{col_widths[j]}}'
                                for j in range(len(row)))  # 左对齐每列
            x_file.write(row_str + '\n')
            
        
        # 绘制混淆矩阵
        plt.figure(figsize=(16, 10))
        
        
        # 创建掩码隐藏0值的单元格
        mask = np.array(confusion) == 0  # 生成True/False矩阵
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",mask=mask,
                    annot_kws={
                        "fontsize": 12,       # 增大注释字体
                        "weight": "bold"      # 加粗字体
                    },
                    xticklabels=classname,
                    yticklabels=classname)
        
        
        # 设置坐标轴字体
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.title('Confusion Matrix', fontsize=14)

        # 设置刻度标签字体大小
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.tight_layout()
        plt.savefig(file_name+'confusion_matrix.png')
        # plt.show()
    
    # 构建返回结果
    result = classification + '\n{} OA(%)'.format(oa * 100) + '\n{} AA (%)'.format(aa * 100)
    if flops and params:
        result += f'\nModel FLOPs: {flops}'
        result += f'\nModel Parameters: {params}'
    if avg_inference_time:
        result += f'\nAverage Inference Time per Sample: {avg_inference_time:.4f} ms'
    
    return result


'------------测试训练好的模型----------------------'


def TestModel(net, DsetLoader,model_path,classname,saveE=False, flops=None, params=None):
    ######对所有数据进行分类预测############
    net.eval()  # 注意启用测试模式
    count = 0
    total_inference_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels, fpath in tqdm(DsetLoader):
            inputs, labels = inputs.cuda(), labels.cpu().numpy()
            
            # 计算推理时间
            start_time = time.time()
            outputs = net(inputs)
            inference_time = time.time() - start_time
            
            total_inference_time += inference_time
            total_samples += inputs.shape[0]
            
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred = outputs
                y = labels
                count = 1
            else:
                y_pred = np.concatenate((y_pred, outputs))
                y = np.concatenate((y, labels))
                
            # 记录错分类的点云
            if saveE:
                # 创建保存目录
                misclassified_dir = os.path.join(os.path.dirname(model_path), 'misclassified')
                os.makedirs(misclassified_dir, exist_ok=True)
                for i in range(len(labels)):
                    if outputs[i] != labels[i]:
                        points = inputs[i].cpu().numpy()
                        save_path = os.path.join(
                            misclassified_dir,
                            f'{os.path.basename(fpath[i])}_pred_{classname[outputs[i]]}.png'
                        )
                        visualize_misclassified_point_cloud(
                            points, labels[i].item(), outputs[i].item(), classname,
                            save_path)
    
    # 计算单个实例的平均推理时间
    avg_inference_time = total_inference_time / total_samples * 1000  # 转换为毫秒
    print(f"\nAverage inference time per sample: {avg_inference_time:.4f} ms")
    print(f"Model FLOPs: {flops}")
    print(f"Model Parameters: {params}")
    
    return SaveTestReport(y_pred, y, model_path[:-4] , classname, flops=flops, params=params, avg_inference_time=avg_inference_time)



def TestModelMain(model_path):
    '''HYPER PARAMETER'''
    
    '------------模型测试----------------------'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '------------生成报告----------------------'
    dl=" "  # 根据数据集选择分隔符
    checkpoint = torch.load(model_path)

    DataPath = checkpoint['datapath'].split('Pts\\')[-1]  # 获取数据集路径
    classnamepath = DataPath + '\modelnet40_shape_names.txt'
    with open(classnamepath, 'r') as f:
        classname = f.read().splitlines()
    args=checkpoint["config"]
    logger = logging.getLogger(__name__)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger.info(args)

    # print(args.pretty())
    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path(DataPath)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH,
                                      npoint=args.num_point,
                                      split='test',
                                      normal_channel=args.normal,
                                      dl=dl)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
    '''MODEL LOADING'''
    args.input_dim = 6 if args.normal else 3
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)),'PointTransformerCls')(args).cuda()  #根据配置文件中选择的model导入相应的分类网络结构 并实例化传入args
    classifier.load_state_dict(checkpoint['model_state_dict'])  #载入模型权重给分类模型
    
    # 计算模型FLOPs
    classifier.eval()
    input_example = torch.randn(1, args.num_point, args.input_dim).cuda()
    flops, params = profile(classifier, inputs=(input_example,))
    
    # 强制转换为G单位
    flops_g = flops / 1e9  # 1G = 1e9 FLOPs
    flops = f"{flops_g:.3f}G"  # 格式化为3位小数的G单位
    
    # 参数处理：先获取数值，再格式化
    params_value = params / 1e6
    params_unit = "M"  # 默认使用M单位
 
 
    
    # 格式化为3位小数
    params = f"{params_value:.3f}{params_unit}"
    
    logger.info(f"Model FLOPs: {flops}, Parameters: {params}")
    
    logger.info('Use pretrain model to Test')
    logger.info(TestModel(classifier, DsetLoader=testDataLoader,model_path=model_path, classname=classname, flops=flops, params=params))


if __name__ == "__main__":
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
    TestModelMain(ModelPATH)
