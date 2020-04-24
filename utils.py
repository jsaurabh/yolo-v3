#utils
from typing import List, Union
import torch
import cv2 as cv
import numpy as np
from pathlib import Path

def read_line(lines: List) -> List:
    """
    Remove spacing, comments and whitespaces from lines

    Args:
        lines (List): list of lines from .cfg

    Returns:
        List of lines of cleaned input 
    """
    config = [c for c in lines if c]
    config = [c for c in config if c[0] != "#"]
    config = [c.rstrip().lstrip() for c in config]
    return config

def parse_blocks(lines: List) -> List:
    """
    Parse list of blocks into singular data structure

    Args:
        lines (List): list of cleaned lines
    
    Returns:
        List of dictionaries, each dictionary a layer in the darknet
    """
    blocks = []
    block = {}

    for line in lines:
        if line[0] == "[":
            if len(block):
                blocks.append(block)
                block = {}
            block['arch'] = line[1:-1].rstrip()
        else:
            k,v = line.split("=")
            block[k.rstrip()] = v.lstrip()
    blocks.append(block)
    return blocks


def predict_transform(pred, in_dim, anchors, num_classes):
    """
    """
    batch = pred.size(0)
    stride = in_dim // pred.size(2)
    grid_size = in_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    pred = pred.view(batch, bbox_attrs * num_anchors, grid_size*grid_size)
    pred = pred.transpose(1, 2).contiguous()

    pred = pred.view(batch, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    pred[:,:,0] = torch.sigmoid(pred[:,:,0])
    pred[:,:,1] = torch.sigmoid(pred[:,:,1])
    pred[:,:,4] = torch.sigmoid(pred[:,:,4])
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    pred[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4]) * anchors
    pred[:,:,5:5 + num_classes] = torch.sigmoid(pred[:,:,5:5 + num_classes])
    pred[:,:,:4] *= stride
    
    return pred

def get_input(img: Union[Path, str]) -> torch.Tensor:
    image = cv.imread('giraffe.png')
    image = cv.resize(image, (608, 608))
    image = image[:,:,::-1].transpose((2, 0, 1))
    image = image[np.newaxis,:,:,:]/255.0
    image = torch.from_numpy(image).float()
    return image.clone().detach()
