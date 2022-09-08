'''
Date: 2021-12-30 4:07:36 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2021-12-30 4:10:53 pm
LastEditors: dihuangdh
'''

import cv2
from cv2 import data
import numpy as np
import torch
import random


rgb_dict = {
    0: (255, 153, 255),
    1: (255, 153, 153),
    2: (255, 102, 102),
    3: (255, 51, 51),
    4: (255, 0, 0),
    5: (102, 255, 102),
    6: (51, 255, 51),
    7: (0, 255, 0),
    8: (255, 204, 204),
    9: (255, 178, 102),
    10: (255, 153, 51),
    11: (255, 128, 0),
    12: (153, 255, 153),
    13: (102, 178, 255),
    14: (51, 153, 255),
    15: (0, 128, 255),
    16: (255, 204, 153),
    17: (255, 102, 255),
    18: (255, 51, 255),
    19: (255, 0, 255),
    20: (153, 204, 255),
}

parent_dict = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 0,
    6: 5,
    7: 6,
    8: 7,
    9: 0,
    10: 9,
    11: 10,
    12: 11,
    13: 0,
    14: 13,
    15: 14,
    16: 15,
    17: 0,
    18: 17,
    19: 18,
    20: 19,
}


class Visualizer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def visualize_kp2d(img, kp2d, score, score_thr=0.4):
        line_width = 3
        circle_rad = 3

        img = cv2.UMat(img)

        for i in range(21):
            current = i
            parent = parent_dict[i]

            kps_cur = (kp2d[current][0].astype(np.int32), kp2d[current][1].astype(np.int32))
            kps_par = (kp2d[parent][0].astype(np.int32), kp2d[parent][1].astype(np.int32))

            if score[current] > score_thr and score[parent] > score_thr and parent != -1:
                img = cv2.line(img, kps_cur, kps_par, rgb_dict[parent][::-1], line_width-1)
            if score[current] > score_thr:
                img = cv2.circle(img, kps_cur, circle_rad, rgb_dict[current][::-1], -1)
            if score[parent] > score_thr and parent != -1:
                img = cv2.circle(img, kps_par, circle_rad, rgb_dict[parent][::-1], -1)
        
        return cv2.UMat.get(img)
    
    @staticmethod
    def visualize_det_kp2ds(img, det, kp2ds):
        # visualize
        img_viz = img.copy()[..., ::-1]
        img_viz = np.ascontiguousarray(img_viz)
        label = None
        Visualizer.plot_one_box(det[0], img_viz, label=label, color=[255, 0, 0], line_thickness=3)
        Visualizer.plot_one_box(det[1], img_viz, label=label, color=[0, 255, 0], line_thickness=3)
        img_viz = Visualizer.visualize_kp2d(img_viz, kp2ds[0, :, :2], kp2ds[0, :, -1], score_thr=0.)
        img_viz = Visualizer.visualize_kp2d(img_viz, kp2ds[1, :, :2], kp2ds[1, :, -1], score_thr=0.)
        
        return img_viz
    
    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        # import pdb;pdb.set_trace()
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def visualize_2d_training_batch(self, data, num=3):
        imgs = []
        for bs in range(min(data['img'].size(0), num)):
            # plot ground-truth 2d
            img = data['img'][bs].clone().cpu().numpy().astype(np.float32)
            kp3d_coord = data['kp3d_coord'][bs].clone().cpu().numpy().astype(np.float32)

            img *= 255.; img = np.transpose(img, (1, 2, 0))
            img = self.visualize_kp2d(img, kp3d_coord[:, :3], kp3d_coord[:, -1], score_thr=0.)
            img = np.transpose(img, (2, 0, 1)); img /= 255.
            img = torch.from_numpy(img)  # (3, 256, 256)
            img_gt = img.clone()

            # plot pred 2d
            if 'kp3d_coord_pred' in data.keys():
                img = data['img'][bs].clone().cpu().numpy().astype(np.float32)
                kp3d_coord_pred = data['kp3d_coord_pred'][bs].clone().detach().cpu().numpy().astype(np.float32)
                img *= 255.; img = np.transpose(img, (1, 2, 0))
                img = self.visualize_kp2d(img, kp3d_coord_pred[:, :3], kp3d_coord_pred[:, -1], score_thr=0.)
                img = np.transpose(img, (2, 0, 1)); img /= 255.
                img = torch.from_numpy(img)  # (3, 256, 256)
                img_pred = img.clone()
            elif 'kp2d_coord_pred' in data.keys():
                img = data['img'][bs].clone().cpu().numpy().astype(np.float32)
                kp2d_coord_pred = data['kp2d_coord_pred'][bs].clone().detach().cpu().numpy().astype(np.float32)
                img *= 255.; img = np.transpose(img, (1, 2, 0))
                img = self.visualize_kp2d(img, kp2d_coord_pred[:, :2], kp2d_coord_pred[:, -1], score_thr=0.)
                img = np.transpose(img, (2, 0, 1)); img /= 255.
                img = torch.from_numpy(img)  # (3, 256, 256)
                img_pred = img.clone()
            else:
                raise ValueError()

            img = torch.cat([img_gt, img_pred], -1)
            imgs.append(img)
        
        imgs = torch.cat(imgs, 1)
        return [imgs]
    
    def visualize_hm_training_batch(self, data, num=3):
        heatmaps = []
        for bs in range(min(data['img'].size(0), num)):
            # plot ground-truth heatmap
            hm2d = data['hm2d'][bs].clone().cpu()
            hm2d = hm2d.sum(0)
            hm2d = torch.clamp(hm2d, 0, 1)[None, :, :].repeat(3, 1, 1)
            hm1d = data['hm1d'][bs].clone().cpu()
            hm1d = hm1d.sum(0)  # (64)
            hm1d = torch.clamp(hm1d, 0, 1)[None, None, :].repeat(3, 10, 1)
            hm2d = torch.cat([hm2d, hm1d], 1)

            # plot pred heatmap
            hm2d_pred = data['hm2d_pred'][bs].clone().cpu()
            hm2d_pred = hm2d_pred.sum(0)
            hm2d_pred = torch.clamp(hm2d_pred, 0, 1)[None, :, :].repeat(3, 1, 1)
            hm1d_pred = data['hm1d_pred'][bs].clone().cpu()
            hm1d_pred = hm1d_pred.sum(0)
            hm1d_pred = torch.clamp(hm1d_pred, 0, 1)[None, None, :].repeat(3, 10, 1)
            hm2d_pred = torch.cat([hm2d_pred, hm1d_pred], 1)

            heatmap = torch.cat([hm2d, hm2d_pred], -1)
            heatmaps.append(heatmap)
        
        heatmaps = torch.cat(heatmaps, 1)
        return [heatmaps]
    
    def visualize_hm2d_training_batch(self, data, num=3):
        heatmaps = []
        for bs in range(min(data['img'].size(0), num)):
            # plot ground-truth heatmap
            hm2d = data['hm2d'][bs].clone().cpu()
            hm2d = hm2d.sum(0)
            hm2d = torch.clamp(hm2d, 0, 1)[None, :, :].repeat(3, 1, 1)

            # plot pred heatmap
            hm2d_pred = data['hm2d_pred'][bs].clone().cpu()
            hm2d_pred = hm2d_pred.sum(0)
            hm2d_pred = torch.clamp(hm2d_pred, 0, 1)[None, :, :].repeat(3, 1, 1)

            heatmap = torch.cat([hm2d, hm2d_pred], -1)
            heatmaps.append(heatmap)
        
        heatmaps = torch.cat(heatmaps, 1)
        return [heatmaps]