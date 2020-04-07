# imports
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import read_line, parse_blocks

def parse_config(cfg):
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

class DetectionLayer(nn.Module):
    def __init__(self, anchor):
        super(DetectionLayer, self).__init__()
        self.anchors = anchor

class Empty(nn.Module):
    def __init__(self):
        super(Empty, self).__init__()

def construct(blocks: List) -> Tuple[dict, torch.nnModuleList]:
    moduleList = nn.ModuleList()

    for idx, layer in enumerate(blocks[1:]):
        module = nn.Sequential()
        output_filters = []
        prev_filters = 3

        if layer['arch'] == 'convolutional':
            activation = layer['activation']
            filters = layer['filters']
            stride = layer['stride']
            kernel = layer['size']
            padding = layer['pad']

            if padding:
                pad = (kernel - 1) // 2
            try:
                batch_norm = layer['batch_normalize']
                bias = False
            except:
                batch_norm = 0
                bias = True
            
            conv = nn.Conv2D(prev_filters, filters, kernel, stride, pad, bias = bias)
            moduleList.add_module(f"conv_{idx}", conv)

            if batch_norm:
                moduleList.add_module(f"bn_{idx}", nn.BatchNorm2D(filters))
            
            if activation == "leaky":
                moduleList.add_module(f"leaky_{idx}", nn.LeakyReLU(0.1, inplace = True))
        
        elif layer['arch'] == 'upsample':
            stride = layer['stride']
            upsample = nn.UpSample(scale_factor = 2, mode = 'bilinear')
            module.add_module(f"upsample_{idx}", upsample)

        elif layer['arch'] == 'shortcut':
            empty = Empty()
            module.add_module(f"shortcut_{idx}", empty)
        
        elif layer['arch'] == 'route':
            l = layer['layers'].split(',')
            start = l[0]
            try:
                end = l[1]
            except:
                end = 0
            
            if start > 0:
                start -= idx
            if end > 0:
                end -= idx
            
            route = Empty()
            module.add_module(f"route_{idx}", route)
            
            if end < 0:
                filters = output_filters[start + idx] + output_filters[idx + end]
            else:
                filter = output_filters[start + idx]
            
        elif layer['arch'] == 'yolo':
            mask = layer['mask'].split(',')
            mask = [int(m) for m in mask]

            anchors = layer['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchor = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchor = [anchor[m] for m in mask]

            detect = DetectionLayer(anchor)
            module.add_module(f"Detection_{idx}", detect)
    
        moduleList.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (blocks[0], moduleList)