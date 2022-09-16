#!/usr/bin/env python
from __future__ import print_function
import argparse
from bdb import set_trace
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2
from zeit.easymocap.camera_utils import UndistortFisheye
from zeit.visualizers.kp2dvis import Visualizer
from zeit.easymocap.triangulation import triangulate
from zeit.easymocap.triangulation import projectN3
from zeit.filters.oneeuro import OneEuroFilter
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
USE_MULTI_GPU = False



# 检测机器是否有多张显卡

if USE_MULTI_GPU and torch.cuda.device_count() > 1:

    MULTI_GPU = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = " 1,2,3"

    device_ids = [1,2,3]

else:

    MULTI_GPU = False
    device_ids = [0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


writer = SummaryWriter("./Log")
# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#     def forward(self, x, target, smoothing=0.1):
#         confidence = 1. - smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = confidence * nll_loss + smoothing * smooth_loss
#         return loss.mean()
class HandVisualizer:
    def __init__(self) -> None:
        plt.ion()
        self.first=1
        self.links=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    def set_axes_equal(self,ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    def show(self,kp3d):
        # (42,4)
        assert len(kp3d)==42
        
        if self.first!=1:
            plt.clf()
        else:
            def getLen(xs):
                return max(xs)-min(xs)
            def getCen(xs):
                return (max(xs)+min(xs))/2
            mask=kp3d[:,3]>0
            kp=kp3d[mask]
            self.box_l=max(getLen(kp[:,0]),getLen(kp[:,1]),getLen(kp[:,2]))
            self.box_cen=[getCen(kp[:,0]),getCen(kp[:,1]),getCen(kp[:,2])]
            self.box_l*=1
        self.first=0
        ax = plt.axes(projection='3d')
        def show_onehand(ps):
            for l in self.links:
                mask=ps[l][:,3]>0
                p=ps[l][mask]
                x=p[:,1]
                y=p[:,0]
                z=p[:,2]
                # print(mask)
                c= np.array([i*5 for i in l])[mask]
                # ax.scatter3D(x, y, z, c=c)
                ax.plot3D(x, y, z )
                ax.set_title('3d Scatter plot')
        def add_box():
            x=[self.box_cen[1]-self.box_l ,self.box_cen[1]+self.box_l]
            y=[self.box_cen[0]-self.box_l ,self.box_cen[0]+self.box_l]
            z=[self.box_cen[2]-self.box_l ,self.box_cen[2]+self.box_l]
            ax.scatter(x,y,z)
        
        show_onehand(kp3d[:21])
        show_onehand(kp3d[21:])
        add_box()


        self.set_axes_equal(ax)
        plt.show()
        plt.pause(0.1)
hv=HandVisualizer()        
def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--keep_rate',
        type=float,
        default=0.9,
        help='keep probability for drop')
    parser.add_argument(
        '--groups',
        type=int,
        default=8,
        help='decouple groups')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./save_models/" + arg.Experiment_name
        arg.work_dir = "./work_dir/" + arg.Experiment_name
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input(
                            'Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        """ 
            加载数据
        """
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        # train的dataloader
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size*len(device_ids),
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        # test的dataloader
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)  
        
        # import pdb;pdb.set_trace()
    def load_model(self):
        """ 
            加载模型
        """
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        # self.loss = LabelSmoothingCrossEntropy().cuda(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
                # model = MyNetwork().to(output_device) #实例化模型并加载到cpu货GPU中
                # model.load_state_dict(torch.load(self.arg.weights))  #加载模型参数，model_cp为之前训练好的模型参数（zip格式）
                # #重新保存网络参数，此时注意改为非zip格式
                # torch.save(model.state_dict(), model_cp,_use_new_zipfile_serialization=False)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        # 没有多设备gpu
        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        # 模型转换为gpu并行
        if MULTI_GPU:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        # optimizer转换为gpu并行
        

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)
        if MULTI_GPU:

                    self.optimizer = nn.DataParallel(self.optimizer, device_ids=device_ids)
                    self.lr_scheduler = nn.DataParallel(self.lr_scheduler, device_ids=device_ids)
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir + '/eval_results')

        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            if MULTI_GPU:
                for param_group in self.optimizer.module.param_groups:
                    param_group['lr'] = lr
                return lr
            else:            
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        # 训练
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        # 获取下train的dataloader
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if epoch >= self.arg.only_train_epoch:
            print('only train part, require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            print('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')
        # 加载数据进行训练
        for batch_idx, (data, label, index) in enumerate(process):
            
            self.global_step += 1

            # todo:将input转换为figure图像进行tensroboard的保存
            # grid = torchvision.utils.make_grid(inputs)
            
            # # width, height
            # fig = plt.figure(figsize=(1 * 2.5, 3), dpi=100)
            # # 将图像还原至标准化之前
            # # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            # npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            # plt.imshow(npimg.astype('uint8'))
            # # title = "{}, {:.2f}%\n(label: {})".format(
            # #     flower_class[str(preds[batch_idx])],  # predict class
            # #     probs[batch_idx] * 100,  # predict probability
            # #     flower_class[str(label[batch_idx])]  # true class
            # # )
            # # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
            # ax = fig.add_subplot(1, 1, batch_idx+1, xticks=[], yticks=[])
            # # ax.set_title(title, color=("green" if preds[batch_idx] == label[batch_idx] else "red"))
            # writer.writer.add_figure('inputs', inputs, batch_idx)

            # get data
            data = Variable(data.float().cuda(
                self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(
                self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            if epoch < 100:
                keep_prob = -(1 - self.arg.keep_rate) / 100 * epoch + 1.0
            else:
                keep_prob = self.arg.keep_rate

    
            output = self.model(data, keep_prob)
            # writer.add_graph(self.model, data.detach())
            # writer.add_histogram("conv1",self.model.conv1.weight,batch_idx)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            # 通过输出 label l1来求loss
            # CrossEntropyLoss
            # GT label
            loss = self.loss(output, label) + l1

            


            self.optimizer.zero_grad()
            loss.backward()
            if MULTI_GPU:
                self.optimizer.module.step()
            else:
                self.optimizer.step()
            loss_value.append(loss.data)
            timer['model'] += self.split_time()

        
            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())

            if MULTI_GPU:
                self.lr = self.optimizer.module.param_groups[0]['lr']
            else:
                self.lr = self.optimizer.param_groups[0]['lr']
            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx, len(loader), loss.data, self.lr))
            timer['statistics'] += self.split_time()
            
            writer.add_scalar('Loss/train', loss.data, batch_idx)
            writer.add_scalar('lr/train', self.lr, batch_idx)
            # 每个batch size保存模型参数
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])
            # import pdb;pdb.set_trace()
            torch.save(weights, self.arg.model_saved_name +
                    '-' + str(batch_idx) + '.pt')
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in state_dict.items()])

        torch.save(weights, self.arg.model_saved_name + 'epoch'+

                   '-' + str(epoch) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        with torch.no_grad():
            self.print_log('Eval epoch: {}'.format(epoch + 1))
            for ln in loader_name:
                loss_value = []
                score_frag = []
                right_num_total = 0
                total_num = 0
                loss_total = 0
                step = 0
                # 获取dataloader来进行process
                process = tqdm(self.data_loader[ln])
                # 从dataloader里面取出
                # batch_idx
                # data 数据
                # label 标签
                # index index
                for batch_idx, (data, label, index) in enumerate(process):
                    # 显示3维数据的手势
                    # data[0,:,0,:,:].shape
                    # torch.Size([3, 27, 1])
                    # data.shape
                    # torch.Size([64, 3, 160, 27, 1])
                    data_tmp = data[0,:,0,:,:]
                    img = np.zeros([400, 640, 3])
                    dets = np.zeros([2, 6])
                    keypoints3d = np.zeros([2, 21, 3])
                    keypoints3d[0] = data_tmp.permute(2,1,0).numpy()[:,0:21,:]
                    img_det = Visualizer.visualize_det_kp2ds(img, dets, keypoints3d)
                    keypoints3d_tmp = np.zeros([42, 4])
                    keypoints3d_tmp[:21,:3]=keypoints3d[0,:,:]
                    keypoints3d_tmp[21:,:3]=keypoints3d[1,:,:]
                    keypoints3d_tmp[:,3]=1
                    print(keypoints3d[:,-1])
                    # hv.show(keypoints3d_tmp)

                    # cv2.imshow("data_tmp",img_det)
                    # key = cv2.waitKey(1)
                    # if key & 0xFF == ord('q'):
                    #     break
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False)
                    # run model
                    with torch.no_grad():
                        output = self.model(data)

                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    # 从预测值和label GT计算相应的loss值 
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.cpu().numpy())

                    # 下一步
                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' +
                                        str(x) + ',' + str(true[i]) + '\n')
                score = np.concatenate(score_frag)

                if 'UCLA' in arg.Experiment_name:
                    self.data_loader[ln].dataset.sample_name = np.arange(
                        len(score))
                # topk求accuracy
                accuracy = self.data_loader[ln].dataset.top_k(score, 1)
                writer.add_scalar('lr/accuracy', accuracy, batch_idx)
                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    score_dict = dict(
                        zip(self.data_loader[ln].dataset.sample_name, score))

                    with open('./work_dir/' + arg.Experiment_name + '/eval_results/best_acc' + '.pkl'.format(
                            epoch, accuracy), 'wb') as f:
                        pickle.dump(score_dict, f)

                print('Eval Accuracy: ', accuracy,
                    ' model: ', self.arg.model_saved_name)

                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))
                self.print_log('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)))
                for k in self.arg.show_topk:
                    self.print_log('\tTop{}: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

                with open('./work_dir/' + arg.Experiment_name + '/eval_results/epoch_' + str(epoch) + '_' + str(accuracy) + '.pkl'.format(
                        epoch, accuracy), 'wb') as f:
                    pickle.dump(score_dict, f)
        return np.mean(loss_value)
    def start(self):
        # 训练
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = int(self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size)
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)
                # 训练
                self.train(epoch, save_model=save_model)
                # test评估暂时未修改
                # 使用test来评估
                val_loss = self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

                self.lr_scheduler.step(val_loss)

            print('best accuracy: ', self.best_acc,
                  ' model_name: ', self.arg.model_saved_name)
        # 测试
        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=self.arg.start_epoch, save_score=self.arg.save_score,
                      loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

from yaml import Loader, Dumper
if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f,Loader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
    writer.close()    
