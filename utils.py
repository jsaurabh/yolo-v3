#utils
from typing import List

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