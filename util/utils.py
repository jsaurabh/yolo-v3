#utils
from typing import List, Union
import torch
import cv2 as cv
import numpy as np
from pathlib import Path
from torch import tensor

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

def load_classes(classfile):
    f = open(classfile, 'r')
    names = f.read().split("\n")[:-1]
    return names
    
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
    image = cv.imread('images/' + img)
    image = cv.resize(image, (608, 608))
    image = image[:,:,::-1].transpose((2, 0, 1))
    image = image[np.newaxis,:,:,:]/255.0
    image = torch.from_numpy(image).float()
    return image.clone().detach()

def unique(t):
    nump = t.cpu().numpy()
    unique = np.unique(nump)
    unique = torch.from_numpy(unique)
    res = tensor.new(unique.shape)
    res.copy_(unique)
    return res

def ious(box1, box2):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def display(pred, confidence, num_classes, nms_conf = 0.4):
    """
    """
    write = 0
    confidence_mask = (pred[:,:,4] > confidence).float().unsqueeze(2)
    pred *= confidence_mask

    corners = pred.new(pred.shape)
    corner[:,:,0] = (pred[:,:,0] - pred[:,:,2])/2
    corner[:,:,1] = (pred[:,:,1] - pred[:,:,3])/2
    corner[:,:,2] = (pred[:,:,0] - pred[:,:,2])/2
    corner[:,:,3] = (pred[:,:,1] - pred[:,:,3])/2
    pred[:,:,:4] = corners[:,:,:4]

    batch = pred.size(0)

    for idx in range(batch):
        prd = pred[idx]
        max_c, max_c_score = torch.max(prd[:,5:5+ num_classes], 1)
        max_confidence = max_c_score.float.unsqueeze(1)
        max_c = max_c.float().unsqueeze(1)

        seq = (prd[:,:5], max_c, max_confidence)
        prd = torch.cat(seq, 1)

        non_zero_idx = torch.nonzero(prd[:,4])
        try:
            prd = prd[non_zero_idx.squeeze(), :].view(-1, 7)
        except:
            continue

        if prd.shape[0] == 0:
            continue

        classes = unique(prd[:, -1])
        for cls in classes:
            mask = prd*(prd[:,-1] == cls).float().unsqueeze(1)
            cls_mask_idx = torch.nonzero(mask[:,-2]).squeeze()
            prd_cls = prd[cls_mask_idx].view(-1,7)

            confidence_sort_idx = torch.sort(prd_cls[:,4], descending = True )[1]
            prd_cls = prd_cls[confidence_sort_idx]
            index = prd_cls.size(0)

            for i in range(index):
                try:
                    iou = ious(prd_cls[i].unsqueeze(0), prd_cls[i+1:])  
                except:
                    break  
                    
                iou_mask = (iou < nms_conf).float().unsqueeze(1)
                prd_cls[i+1:] *= iou_mask

                non_zero_idx = torch.nonzero(prd_cls[:,4]).squeeze()
                prd_cls = prd_cls[non_zero_idx].view(-1, 7)


            batch_idx = prd_cls.new(prd_cls.size(0), 1).fill_(idx)
            seq = batch_idx, prd_cls

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                output = torch.cat((output, torch.cat(seq, 1)))

    try:
        return output
    except:
        return 0
    