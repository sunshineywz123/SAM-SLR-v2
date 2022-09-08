'''
Date: 2021-10-20 11:47:11 am
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2021-11-29 3:35:00 pm
LastEditors: dihuangdh
'''
import os
import os.path as osp
import json
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from easymocap.mytools.camera_utils import read_camera


def read_json(jsonname):
    with open(jsonname) as f:
        data = json.load(f)
    return data



root_path = '/home/SENSETIME/zhangchaozhe/Desktop/data/SenseBee'


img_id = 0; annot_id = 0;
for data_split in (('train'),):
    images = []; annotations = [];
    captures = glob(osp.join(root_path, '*'))
    captures.sort()
    intri_path = osp.join(root_path, 'intri.yml')
    extri_path = osp.join(root_path, 'extri.yml')
    cams = read_camera(intri_path, extri_path)
    output={}
    for capture in tqdm(captures):
        if not osp.isdir(capture):
            continue
        # read camera
        # subs = os.listdir(osp.join(capture, 'images'))
        subs = ['cam0', 'cam1']

        subs.sort()
        for sub in tqdm(subs):
            cam = cams[sub[-1]]
            imgnames = os.listdir(osp.join(capture,'images',sub, 'data' ))
            imgnames.sort()
            for imgname in tqdm(imgnames):
                imgpath = osp.join(capture,'images',sub, 'data' ,imgname)
                annot_path = osp.join(capture,'images', sub,'annots',imgname+'.json')
                assert osp.exists(imgpath)
                assert osp.exists(annot_path)

                annot = read_json(annot_path)
                if 'step_1' not in annot:
                    continue
                if 'result' not in annot['step_1']:
                    continue
                result=annot['step_1']['result']
                
                order_range=set(range(1,43))
            

                k2d=[[r['x'],r['y']] for r in result if r['order'] in order_range ]
                if len(k2d) !=len(order_range):
                    continue
                valids=[r['valid'] for r in result if r['order'] in order_range ]

                joint_img = np.array(k2d)

                
                def get_doh_dict(joint_img,hand_size):
                    doh_dict={}
                    w=annot['width']
                    h=annot['height']
                    doh_dict['width']=w
                    doh_dict['height']=h
                    doh_dict['x1']=(joint_img[0:21, 0].min()-10)/w
                    doh_dict['y1']=(joint_img[0:21, 1].min()-10)/h
                    doh_dict['x2']=(joint_img[0:21, 0].max()+10)/w
                    doh_dict['y2']=(joint_img[0:21, 1].max()+10)/h
                    doh_dict['hand_side']=hand_size
                    return doh_dict
                
                doh=[get_doh_dict(joint_img[0:21],'l'),get_doh_dict(joint_img[21:42],'r')]

                output[osp.join(osp.basename(capture),'images',sub, 'data'  ,imgname)]=doh

                    # break
    

    output_path = osp.join(root_path, 'SenseBee_detection_' + data_split + '.json')
    with open(output_path, 'w') as f:
        json.dump(output, f)
    print('Save at ' + output_path)
                
