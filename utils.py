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
            block['arch'] = line.replace("[", "").replace("]", "")
        else:
            k,v = line.split("=")
            block[k] = v
    blocks.append(block)
    return blocks

def predict_transform(pred, anchors, in_dim, num_classes):
    """
    """
    batch = pred.size(0)
    stride = in_dim // pred.size(2)
    grid_size = in_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    pred = pred.view(batch, bbox_attrs * num_anchors, grid_size**2).transpose(1, 2).contiguous()
    pred = pred.view(batch, (grid_size ** 2) * num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    pred[:,:,0] = torch.sigmoid(pred[:,:,0])
    pred[:,:,1] = torch.sigmoid(pred[:,:,1])
    pred[:,:,4] = torch.sigmoid(pred[:,:,4])

    a, b = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    print(a.shape, b.shape)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    print(x_offset.shape, y_offset.shape)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    pred[:,:,:2] += x_y_offset

    anchors = torch.Tensor(anchors)
    anchors.repeat(grid_size ** 2, 1).unsqueeze(0)
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4]) * anchors
    pred[:,:,5:5 + num_classes] = torch.sigmoid(pred[:,:,5:5 + num_classes])
    pred[:,:,:4] *= stride
    
    return pred

def get_input(img: Union[Path, str]) -> torch.Tensor:
    image = cv.imread('giraffe.png')
    image = cv.resize(image, (416, 416))
    image = image[:,:,::-1].transpose((2, 0, 1))
    image = image[np.newaxis,:,:,:]/255.0
    image = torch.from_numpy(image).float()
    # print(type(torch.tensor(image)))
    return image.clone().detach()