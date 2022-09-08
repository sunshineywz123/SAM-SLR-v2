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

root_path = '/home/SENSETIME/zhangchaozhe/Desktop/data/SenseBee_train'
cnt_debug =0 

img_id = 0; annot_id = 0;
for data_split in (('train'),):
    images = []; annotations = [];
    captures = glob(osp.join(root_path, '*'))
    captures.sort()
    intri_path = osp.join(root_path, 'intri.yml')
    extri_path = osp.join(root_path, 'extri.yml')
    cams = read_camera(intri_path, extri_path)
    output={}
    for capture in captures:
        if not osp.isdir(capture):
            continue
        # read camera
        # subs = os.listdir(osp.join(capture, 'images'))
        subs = ['cam0', 'cam1']

        subs.sort()
        for sub in subs:
            cam = cams[sub[-1]]
            imgnames = os.listdir(osp.join(capture,'images',sub, 'data' ))
            imgnames.sort()
            for imgname in imgnames:
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
                import re
                
                if len(result)<42:
                    continue
                    
                result.sort(key=lambda r : int(r['order']))
                
                
                if result[0]['attribute']=="":
                    order_range=set(range(1,43))
                    k2d=[[r['x'],r['y']] for r in result if r['order'] in order_range ]
                    if len(k2d) !=len(order_range):
                        continue
                else:
                    attrs=[int(re.findall('\d+',r['attribute'])[0]) for r in result]
                    cnt=[2 for i in range(21)]
                    attrWrong=0
                    for idx,a in enumerate(attrs):
                        cnt[a]-=1
                        if cnt[a]==0:
                            if max(cnt)==2:
                                attrWrong|=1
                        if cnt[a]<0:
                            attrWrong|=2
                    # if (attrWrong&2)>0:     
                        
                    #     pass
                    #     cnt_debug+=1
                    #     print(cnt_debug,'wrong attrs',annot_path)
                    #     # print(handle_img_path_tmp(imgpath))
                    # elif (attrWrong&1)>0:
                    #     pass
                    #     # cnt_debug+=1
                    #     # print(annot_path)
                    #     # print(cnt_debug,'wrong attrs',annot_path)
                    #     # print(handle_img_path_tmp(imgpath))

                    if attrWrong>0:
                        continue

                        
                    
                    k2d=[[] for _ in range(42)]


                    # k2d=[[r['x'],r['y']] for r in result ]
                    # if len(k2d) !=len(order_range):
                    #     continue

                    # attrsorted=list(range(0,21))+list(range(0,21))

                    # if attrs!=attrsorted:
                    #     continue

                    cnt=[2 for i in range(21)]
                    for idx,a in enumerate(attrs):
                        cnt[a]-=1
                        if cnt[a]==1:
                            k2d[a]=[result[idx]['x'],result[idx]['y']]
                        else:
                            k2d[a+21]=[result[idx]['x'],result[idx]['y']]
                    
                # valids=[r['valid'] for r in result ]

                joint_img = np.array(k2d)
                img=cv2.imread(imgpath)
                print(img.shape)
                print(annot['width'],annot['height'])
                def get_doh_dict(joint_img,hand_size):
                    doh_dict={}
                    w=annot['width']
                    h=annot['height']
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

                output[osp.join(osp.basename(capture),'images',sub, 'data'  ,imgname)]=doh
                print(osp.join(osp.basename(capture),'images',sub, 'data'  ,imgname))
        # print(capture,len(output))            # break
    

    output_path = osp.join(root_path, 'SenseBee_detection_' + data_split + '.json')
    print(len(output))
    with open(output_path, 'w') as f:
        json.dump(output, f)
    print('Save at ' + output_path)
                
