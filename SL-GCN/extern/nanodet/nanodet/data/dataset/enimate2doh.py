'''
Date: 2021-10-20 11:47:11 am
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2021-11-29 3:35:00 pm
LastEditors: dihuangdh
'''
from cgi import test
import os
import os.path as osp
import json
import cv2
from cv2 import sort
import numpy as np
from tqdm import tqdm
from glob import glob
from easymocap.mytools.camera_utils import read_camera


def read_json(jsonname):
    with open(jsonname) as f:
        data = json.load(f)
    return data


def handle_img_path_tmp(path):
    if 'occlude' in path:
        idx=path.find('occlude')
        s=path[idx:]
        s=s.replace('images/','')
    else:
        idx=path.find('badcase')
        s=path[idx:]
        s=s.replace('images/','')
        s=s.replace('cam','')
        s=s.replace('data/','')
    return s

root_path = '/home/SENSETIME/zhangchaozhe/Desktop/workplace/sense-hand-clean/sense-hand-demo/data/enimate/test/occlude'

images = []; annotations = [];
captures = glob(osp.join(root_path, '*'))
captures.sort()
intri_path = osp.join(root_path, 'intri.yml')
extri_path = osp.join(root_path, 'extri.yml')
cams = read_camera(intri_path, extri_path)
output={}

annot_paths=glob(osp.join(root_path,'annots/0/*'))+glob(osp.join(root_path,'annots/1/*'))
img_paths=glob(osp.join(root_path,'images/0/*'))+glob(osp.join(root_path,'images/1/*'))

annot_paths.sort()
img_paths.sort()

ai=0;ii=0;
def fname(path):
    return osp.basename(path).split('.')[0]

for ai in range(len(annot_paths)):
    while fname(img_paths[ii])!=fname(annot_paths[ai]):
        ii+=1
    img_path=img_paths[ii]
    annot_path=annot_paths[ai]
    data=read_json(annot_path)
    k2d=data['annots']
    img=cv2.imread(img_path)



    joint_img = np.array(k2d)

    def get_doh_dict(joint_img,hand_size):
        doh_dict={}
        w=img.shape[1]
        h=img.shape[0]
        doh_dict['width']=w
        doh_dict['height']=h
        doh_dict['x1']=(joint_img[0:21, 0].min())/w
        doh_dict['y1']=(joint_img[0:21, 1].min())/h
        doh_dict['x2']=(joint_img[0:21, 0].max())/w
        doh_dict['y2']=(joint_img[0:21, 1].max())/h
        doh_dict['hand_side']=hand_size
        return doh_dict

    doh=[get_doh_dict(joint_img[0:21],'l'),get_doh_dict(joint_img[21:42],'r')]


    # if doh[0]['x2']>doh[1]['x1'] and (attrWrong&1)>0:
    #     print(handle_img_path_tmp(imgpath))

    output[img_path]=doh
    print(img_path)

    # print(capture,len(output))            # break


output_path = osp.join(root_path, 'SenseBee_detection_' + 'test' + '.json')
print(len(output))
with open(output_path, 'w') as f:
    json.dump(output, f,indent=2)
print('Save at ' + output_path)
                
