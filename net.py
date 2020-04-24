# Network architecture
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import parse_config, construct
from utils import predict_transform, get_input

class DarkNet(nn.Module):
    def __init__(self, config   ):
        super(DarkNet, self).__init__()
        self.blocks = parse_config(config)
        self.net, self.moduleList = construct(self.blocks)

    def load_weights(self, weights):
        """
        """
        f = open(weights, 'rb')
        header = np.fromfile(f, count = 5, dtype = np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(f, dtype = np.float32)
        ptr = 0
        for idx, module in enumerate(self.moduleList):
            layer = self.blocks[idx+1]['arch']
            if layer == 'convolutional':
                model =  self.moduleList[idx]
                try:
                    batch_norm = int(self.blocks[idx+1]['batch_normalize'])
                except:
                    batch_norm = 0

                conv = model[0]
                if batch_norm:
                    bn = model[1]
                    bn_nel = bn.bias.numel()
                    b = torch.from_numpy(weights[ptr: ptr+bn_nel])
                    ptr += bn_nel

                    w = torch.from_numpy(weights[ptr: ptr+bn_nel])
                    ptr += bn_nel
                    
                    mean = torch.from_numpy(weights[ptr: ptr+bn_nel])
                    ptr += bn_nel

                    variance = torch.from_numpy(weights[ptr: ptr+bn_nel])
                    ptr += bn_nel

                    b = b.view_as(bn.bias.data)
                    w = w.view_as(bn.weight.data)
                    mean = mean.view_as(bn.running_mean)
                    variance = variance.view_as(bn.running_var)

                    bn.bias.data.copy_(b)
                    bn.weight.data.copy_(w)
                    bn.running_mean.data.copy_(mean)
                    bn.running_var.data.copy_(variance)
                
                else:
                    bn_nel = conv.bias.numel()
                    b = torch.from_numpy(weights[ptr: ptr+bn_nel])
                    ptr += bn_nel

                    b = b.view_as(conv.bias.data)
                    conv.bias.data.copy_(b)
                
                w_nel = conv.weight.numel()
                w = torch.from_numpy(weights[ptr:ptr + w_nel])
                w = w.view_as(conv.weight.data)
                conv.weight.data.copy_(w)

    def forward(self, x):
        """
        """
        
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
                layers = module['layers']
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
print("Model initiated\n")
model.load_weights('weights/yolov3.weights')
print("Weights loaded\n")
inp = get_input('giraffe.png')
# print(model)
pred = model(inp)
print(pred.shape)