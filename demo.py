import datetime

import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data

from scipy.io import loadmat
import scipy.io as sio
import TripNet
import numpy as np
import time
import os
from utils import train_patch, setup_seed, output_metric, print_args, train_epoch, valid_epoch, All_patch


# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("GLT")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston','SZU'], default='SZU', help='dataset to use')
parser.add_argument('--num_classes', choices=[11, 6, 15,20], default=20, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patches1', type=int, default=12, help='number1 of patches')#houston10 trento8 muufl6 szu 12
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def train_1times():

    ITER=1
    KAPPAlist = []
    OAlist = []
    AAlist = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((ITER, args.num_classes))
    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')
    for i in range(ITER):
        setup_seed(args.seed)
    # -------------------------------------------------------------------------------
    # prepare data
        if args.dataset == 'Houston':
            DataPath1 = './data/houston/houston_hsi.mat'
            DataPath2 = './data/houston/houston_lidar.mat'
            Data1 = loadmat(DataPath1)['houston_hsi']
            Data2 = loadmat(DataPath2)['houston_lidar']
            LabelPath_10TIMES = './data/houston/Houston_HSI_train_gt_new.mat'
            TrLabel_10TIMES = loadmat(LabelPath_10TIMES)['train']
            TsLabel_10TIMES = loadmat('./data/houston/Houston_HSI_test_gt_new.mat')['test']
            y = sio.loadmat('data/houston/houston_gt.mat')['gt']
        elif args.dataset == 'Muufl':
            DataPath1 = './data/muufl_hsi.mat'
            DataPath2 = './data/muufl_lidar.mat'
            Data1 = loadmat(DataPath1)['muufl_hsi']
            Data2 = loadmat(DataPath2)['muufl_lidar']

            LabelPath_10TIMES = './data/Muufl_HSI_train_gt_new.mat'
            TrLabel_10TIMES = loadmat(LabelPath_10TIMES)['train']
            TsLabel_10TIMES = loadmat('./data/Muufl_HSI_test_gt_new.mat')['test']
            y = sio.loadmat('./data/muufl_gt.mat')['muufl_gt']
        elif args.dataset == 'Trento':
            Data1= sio.loadmat('data/HSI_Trento.mat')['hsi_trento']

            Data2 = sio.loadmat('data/Lidar1_Trento.mat')['lidar1_trento']

            LabelPath_10TIMES = './data/Trento/Trento_HSI_train_gt_new.mat'
            TrLabel_10TIMES = loadmat(LabelPath_10TIMES)['train']
            TsLabel_10TIMES = loadmat('./data/Trento/Trento_HSI_test_gt_new.mat')['test']
            y = sio.loadmat('data/Trento/GT_Trento.mat')['gt_trento']
        elif args.dataset == 'SZU':
            Data1 = sio.loadmat('SZU_hsi_down.mat')['data']
            Data2 = sio.loadmat('SZU_chm_down.mat')['data']
            y = np.transpose(sio.loadmat('SZU_u_ulb.mat')['data'])
            TrLabel_10TIMES = loadmat('SZtr_100.mat')['train_labels']
            TsLabel_10TIMES = loadmat('SZte_100.mat')['test_labels']

        Data1 = Data1.astype(np.float32)

        Data2 = Data2.astype(np.float32)


        patchsize1 = args.patches1  # input spatial size for 2D-CNN
        pad_width1 = np.floor(patchsize1 / 2)
        pad_width1 = int(pad_width1)  # 8

        TrainPatch11, TrainPatch21, TrainLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TrLabel_10TIMES)
        TestPatch11, TestPatch21, TestLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)

        gt = y.reshape(np.prod(y.shape[:2]), )
        gt = gt.astype(int)
        # _, all_indices = sampling1(1, gt)

        # all_indices = [j for j, x in enumerate(gt.ravel().tolist()) if x >= 0]
        #
        # X1_all_data,X2_all_data,_ = All_patch(Data1, Data2, patchsize1, pad_width1,y)
        # torch_dataset_all = Data.TensorDataset(X1_all_data, X2_all_data)
        # all_iter = Data.DataLoader(
        #     dataset=torch_dataset_all,  # torch TensorDataset format
        #     batch_size=64,  # mini batch size
        #     shuffle=False,
        #     num_workers=0,
        # )
        train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainLabel)
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestLabel)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        [m1, n1, l1] = np.shape(Data1)
        Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
        height1, width1, band1 = Data1.shape
        height2, width2, band2 = Data2.shape
        # data size
        print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
        print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
        # -------------------------------------------------------------------------------
        # create model
        model=TripNet.SimpleCNN(args.num_classes, 128, band1, band2, ).cuda()
        # criterion
        criterion = nn.CrossEntropyLoss().cuda()
        loss_fun2 = nn.MSELoss().cuda()
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
        if args.flag_test == 'train':
            BestAcc = 0
            val_acc = []
            print("start training")
            tic = time.time()
            for epoch in range(args.epoches):
                # train model
                model.train()

                train_acc, train_obj, tar_t, pre_t = train_epoch(model,  train_loader, criterion, optimizer)


                OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
                print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                      .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
                scheduler.step()

                if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                    model.eval()
                    tar_v, pre_v,_ = valid_epoch(model, test_loader, criterion)
                    OA2, AA2, Kappa2, CA2 = output_metric(tar_v, pre_v)
                    val_acc.append(OA2)
                    print("Every 5 epochs' records:")
                    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
                    print(CA2)
                    if OA2 > BestAcc:
                        torch.save(model.state_dict(), './sz_.pkl')
                        BestAcc = OA2

            toc = time.time()
            model.eval()
            model.load_state_dict(torch.load('./sz_.pkl'))
            tar_v, pre_v, features = valid_epoch(model, test_loader, criterion,2)


            OA, AA, Kappa, CA = output_metric(tar_v, pre_v)

            print("Final records:")
            print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
            print(CA)
            print("Running Time: {:.2f}".format(toc - tic))
            print("Running Time: {:.2f}".format((toc - tic)/epoch))
            print("**************************************************")
            print("Parameter:")
        else:
            model.eval()
            # model.load_state_dict(torch.load('./sz_.pkl'))
            # generate_pic.generate_png(
            #     all_iter, model, y, args.dataset, 'cuda',
            #     all_indices)  # all_iter, net, gt_hsi, Dataset, device, total_indices,





if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()



