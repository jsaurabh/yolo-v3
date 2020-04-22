# Network architecture
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import parse_config, construct
from utils import predict_transform, get_input

class DarkNet(nn.Module):
    def __init__(self, config):
        super(DarkNet, self).__init__()
        self.blocks = parse_config(config)
        self.net, self.moduleList = construct(self.blocks)
    
    def forward(self, x):
        modules = self.blocks[1:]
        detections = []
        features = {}
        collector = 0

        for idx, module in enumerate(modules):
            layer = (module['arch'])

            if layer == 'convolutional' or layer == 'upsample':
                x = self.moduleList[idx](x)
                features[idx] = x

            elif layer == 'shortcut':
                origin = int(module['from'])
                x = features[idx - 1] + features[idx + origin]
                features[idx] = x

            elif layer == 'route':
                layers = module['layers ']
                layers = [int(l) for l in layers]


                if layers[0] > 0:
                    layers[0] = layers[0] - idx
                
                if len(layers) == 1:
                    x = features[idx + (layers[0])]
                
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - idx

                    m1 = features[idx + layers[0]]
                    m2 = features[idx + layers[1]]
                    x = torch.cat((m1, m2), 1)
                features[idx] = x

            elif layer == 'yolo':
                anchors = self.moduleList[idx][0].anchors
                in_dim = int(self.net['height'])
                num_classes = int(module['classes'])

                x = x.data
                x = predict_transform(x, in_dim, anchors, num_classes)

                if not collector:
                    detections = x
                    collector = 1
                else:
                    detections = torch.cat((detections, x), 1)

                features[idx] = features[idx-1]
            
        try:
            return detections   
        except:
            return 0

model = DarkNet('cfgs/yolov3.cfg')
inp = get_input('s')
# print(model)
pred = model(inp)
# print(pred)