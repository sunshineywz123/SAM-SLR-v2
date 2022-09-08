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

import copy
from dataclasses import dataclass
import warnings

from .coco import CocoDataset
from .xml_dataset import XMLDataset
from .doh import DOHDataset
from .mixed_dataset import MixedDatasets
from .ego_xml_dataset import EGOXMLDataset

def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")

    if type(name) is list:
        names=copy.deepcopy(name)
        img_paths=copy.deepcopy(dataset_cfg.pop("img_path"))
        ann_paths=copy.deepcopy(dataset_cfg.pop("ann_path"))
        dbs=[]
        for na,img_path,ann_path in zip(names,img_paths,ann_paths):
            dataset_cfg["name"]=na
            dataset_cfg["img_path"]=img_path
            dataset_cfg["ann_path"]=ann_path
            dbs.append(build_dataset(dataset_cfg,mode))
        return MixedDatasets(dbs)

    if name == "coco":
        warnings.warn(
            "Dataset name coco has been deprecated. Please use CocoDataset instead."
        )
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "xml_dataset":
        warnings.warn(
            "Dataset name xml_dataset has been deprecated. "
            "Please use XMLDataset instead."
        )
        return XMLDataset(mode=mode, **dataset_cfg)
    elif name == "CocoDataset":
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "XMLDataset":
        return XMLDataset(mode=mode, **dataset_cfg)
    elif name == "DOHDataset":
        return DOHDataset(mode=mode, **dataset_cfg)
    elif name =='EGOXMLDataset':
        return EGOXMLDataset(mode=mode,**dataset_cfg)
    else:
        raise NotImplementedError("Unknown dataset type!")
