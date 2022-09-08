import os
import os.path as osp
import yaml


def read_yaml(path):
    assert osp.exists(path), path
    with open(path) as f:
        data = yaml.safe_load(f)
    return data 