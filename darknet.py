# imports
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import read_line, parse_blocks

def parse_config(cfg) -> List:
    """Parse .cfg config file.

    Args:
        cfg (): config file with .cfg file extension
    

    Returns:
        list: list of dictionaries. Each dictionary describes a single block in
        the architecture.
    """
    lines = []
    with open(cfg, 'r') as config:
        for line in config:
            lines.append(line.strip('\n'))
    config = read_line(lines)

    blocks = parse_blocks(config)
    return blocks

class DetectionLayer(nn.Module):
    def __init__(self, anchor):
        super(DetectionLayer, self).__init__()
        self.anchors = anchor

class Empty(nn.Module):
    def __init__(self):
        super(Empty, self).__init__()

def construct(blocks: List) -> Tuple[dict, torch.nn.ModuleList]:
    moduleList = nn.ModuleList()
    output_filters = []
    prev_filters = 3

    for idx, layer in enumerate(blocks[1:]):
        modules = nn.Sequential()

        if layer['arch'] == 'convolutional':
            activation = layer['activation']
            filters = int(layer['filters'])
            stride = int(layer['stride'])
            kernel = int(layer['size'])
            padding = int(layer['pad'])

            if padding:
                pad = (kernel - 1) // 2
            try:
                batch_norm = int(layer['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True
            
            conv = nn.Conv2d(prev_filters, filters, kernel, stride, pad, bias = bias)
            modules.add_module(f"conv_{idx}", conv)

            if batch_norm:
                modules.add_module(f"bn_{idx}", nn.BatchNorm2d(filters))
            
            if activation == "leaky":
                modules.add_module(f"leaky_{idx}", nn.LeakyReLU(0.1, inplace = True))
        
        elif layer['arch'] == 'upsample':
            stride = layer['stride']
            upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            modules.add_module(f"upsample_{idx}", upsample)

        elif layer['arch'] == 'shortcut':
            empty = Empty()
            modules.add_module(f"shortcut_{idx}", empty)
        
        elif layer['arch'] == 'route':
            l = layer['layers '].split(',')
            start = int(l[0])
            try:
                end = int(l[1])
            except:
                end = 0
            
            if start > 0:
                start -= idx
            if end > 0:
                end -= idx
            
            route = Empty()
            modules.add_module(f"route_{idx}", route)
            
            if end < 0:
                filters = output_filters[start + idx] + output_filters[idx + end]
            else:
                filter = output_filters[start + idx]
            
        elif layer['arch'] == 'yolo':
            mask = layer['mask '].split(',')
            mask = [int(m) for m in mask]
 
            anchors = layer['anchors '].split(',')
            anchors = [int(a) for a in anchors]
            anchor = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchor = [anchor[m] for m in mask]

            detect = DetectionLayer(anchor)
            modules.add_module(f"Detection_{idx}", detect)
    
        moduleList.append(modules)
        prev_filters = filters
        output_filters.append(filters)
    
    return (blocks[0], moduleList)

blocks = parse_config("cfgs/yolov3.cfg")
net, moduleList = construct(blocks)