# coding:utf-8
import cv2
import torch
import numpy as np
from networks.ResUnet import ResUnet
from utils.metrics import calculate_metrics
import argparse
from dataloaders.convert_csv_to_list import convert_labeled_list
import os
import sys, traceback
import datetime
import random
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from train_DG import TrainDG
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform, collate_fn_w_transform

class Inference:
    def __init__(self, config, test_loader):
        # 数据加载
        self.test_loader = test_loader

        # 模型
        self.model = None
        self.backbone = config.backbone
        self.model_type = config.model_type
        self.load_time=config.load_time

        # 路径设置
        self.target = config.Source_Dataset
        self.model_path = "./models/"+config.load_time

        # 其他
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.mode = config.mode
        self.device =torch.device("cuda:0")

        self.build_model()
        self.print_network(self.model)

    def build_model(self):
        if self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=True,
                                 mixstyle_layers=None)
        else:
            raise ValueError('The model type is wrong!')

        checkpoint = torch.load(self.model_path + '/' + 'best-val' + '-' + self.model_type + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of parameters: {}".format(num_params))

    def test(self):
        print("Testing and Saving the results... Domain Generalization Phase")
        print("--" * 15)
        metrics_y = [[], [], [], []]
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                x, y, path = data['data'], data['mask'], data['name']
                x = torch.from_numpy(x).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x, y = x.to(self.device), y.to(self.device)

                seg_logit= self.model(x)
                seg_output = torch.sigmoid(seg_logit)
                metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())

                for i in range(len(metrics)):
                    metrics_y[i].append(metrics[i])


        test_metrics_y = np.mean(metrics_y, axis=1)
        print_test_metric = {}
        for i in range(len(test_metrics_y)):
            print_test_metric[metric_dict[i]] = test_metrics_y[i]

        with open('test_'+self.model_path.split('/')[-2]+'.txt', 'w', encoding='utf-8') as f:
            f.write('Disc Dice\n')
            f.write(str(metrics_y[0])+'\n')  # Disc Dice
            f.write('Cup Dice\n')
            f.write(str(metrics_y[2])+'\n')  # Cup Dice

        print("Test Metrics: ", print_test_metric)
        return print_test_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_DG',
                        help='train_DG/single_test/multi_test')   # choose the mode

    parser.add_argument('--load_time', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='Res_Unet', help='Res_Unet')  # choose the model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')


    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD/Adam/AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='Epoch',
                        help='Cosine/Step/Epoch')   # choose the decrease strategy of lr
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # weight_decay in SGD
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD

    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--Source_Dataset', nargs='+', type=str, default=['Drishti_GS'],
                        help='BinRushed/Magrabia/REFUGE/ORIGA/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=str, default='MESSIDOR_Base1',
                        help='BinRushed/Magrabia/REFUGE/ORIGA/Drishti_GS')

    parser.add_argument('--path_save_result', type=str, default='./results/')
    parser.add_argument('--path_save_model', type=str, default='./models/')
    parser.add_argument('--path_save_log', type=str, default='./logs/')
    parser.add_argument('--dataset_root', type=str, default='/home/daocp01/cx/DataSet/Processed_Fundus_Images_val')
    config = parser.parse_args()

    print('Multi_test for single-source domain generalization...')
    print('Train Source: ' + 'Magrabia')
    print('Loading model: ' + str(config.load_time) + '/' + 'best-val' + '-' + str(config.model_type) + '.pth')
    Disc_Dice, Disc_ASD, Cup_Dice, Cup_ASD = [], [], [], []
    test_datasets = ['BinRushed', 'Magrabia', 'REFUGE', 'ORIGA', 'Drishti_GS']

    test_datasets.remove(config.Source_Dataset[0])

    for target in test_datasets:
        target_test_csv = [target + '_val.csv', target + '_train.csv']
        print(target_test_csv)
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)

        target_valid_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                                 config.image_size, img_normalize=True)
        test_dataloader = DataLoader(dataset=target_valid_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=collate_fn_wo_transform,
                                    num_workers=config.num_workers)

        test = Inference(config, test_dataloader)
        result_dict = test.test()
        Disc_Dice.append(result_dict['Disc_Dice']), Disc_ASD.append(result_dict['Disc_ASD'])
        Cup_Dice.append(result_dict['Cup_Dice']), Cup_ASD.append(result_dict['Cup_ASD'])
    print('Mean Disc Dice:{:.8f} Mean Disc ASD:{:.8f} Mean Cup Dice:{:.8f} Mean Cup ASD:{:.8f}'.format(
        np.mean(Disc_Dice), np.mean(Disc_ASD), np.mean(Cup_Dice), np.mean(Cup_ASD)))
    print('***'*10)
    print('\n')
