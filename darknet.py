# imports
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
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
    print(blocks)
parse_config('cfgs/yolov3.cfg')