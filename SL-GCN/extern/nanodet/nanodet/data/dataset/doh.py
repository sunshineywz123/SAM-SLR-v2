# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
import json

from pycocotools.coco import COCO

from .coco import CocoDataset

from .xml_dataset import CocoXML


class DOHDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(DOHDataset, self).__init__(**kwargs)

    def to_coco(self, ann_path):
        """
        convert 100doh annotations to coco_api
        :param ann_path:
        :return:
        """
        print('loading annotations into memory...')
        tic = time.time()
        dataset = json.load(open(ann_path, 'r'))
        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
        ann_id = 1
        for idx, (file_name, objects) in enumerate(dataset.items()):
            info = {
                "file_name": file_name,
                "height": objects[0]['height'],
                "width": objects[0]['width'],
                "id": idx + 1,
            }
            image_info.append(info)
            for _object in objects:
                category = 'lefthand' if _object['hand_side'] == 'l' else 'righthand'
                if category not in self.class_names:
                    logging.error(
                        "ERROR! {} is not in class_names! "
                        "Pass this box annotation.".format(category)
                    )
                    continue
                for cat in categories:
                    if category == cat["name"]:
                        cat_id = cat["id"]
                xmin = round(_object['x1'] * (_object['width'] - 1))
                ymin = round(_object['y1'] * (_object['height'] - 1))
                xmax = round(_object['x2'] * (_object['width'] - 1))
                ymax = round(_object['y2'] * (_object['height'] - 1))
                w = xmax - xmin
                h = ymax - ymin
                if w < 0 or h < 0:
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(file_name)
                    )
                    continue
                coco_box = [max(xmin, 0), max(ymin, 0), min(w, _object['width'] - 1), min(h, _object['height'] - 1)]
                ann = {
                    "image_id": idx + 1,
                    "bbox": coco_box,
                    "category_id": cat_id,
                    "iscrowd": 0,
                    "id": ann_id,
                    "area": coco_box[2] * coco_box[3],
                }
                annotations.append(ann)
                ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        logging.info(
            "Load {} images and {} boxes".format(len(image_info), len(annotations))
        )
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.to_coco(ann_path)
        self.coco_api = CocoXML(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
