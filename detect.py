
from util.utils import load_classes, prep_image, display
from net import DarkNet

import argparse
import os
import time
import random

import pandas as pd
import torch
import cv2 as cv
import pickle as pkl



def parse():
    p = argparse.ArgumentParser(description = "YOLOv3 Detection")

    p.add_argument("--images", dest = "images", help = "Directory containing images for detection",
                default = "images", type = str)

    p.add_argument("--output", dest = "output", help = "Output directory",
                default = "output", type = str)

    p.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)

    p.add_argument("--conf", dest = "conf", help = "Confidence, to help filter prediction", default = 0.5)

    p.add_argument("--nms", dest = "nms", help = "NMS Threshold", default = 0.4)

    p.add_argument("--cfg", dest = "cfg", help = "Config file path for model",
                default = "cfgs/yolov3.cfg", type = str)

    p.add_argument("--w", dest = "w", help = "Weights file path for model",
                default = "weights/yolov3.weights", type = str)
    
    return p.parse_args()

args = parse()
print("in")
images = args.images
bs = int(args.bs)
confidence = float(args.conf)
nms_threshold = float(args.nms)
start = 0
num_classes = 80

classes = load_classes("data/coco.names")

print("Loading network")
model = DarkNet(args.cfg)
print("Model initiated\n")
model.load_weights(args.w)
print("Weights loaded\n")

model.eval()

try:
    im = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    im = []
    im.append(os.path.join(os.path.realpath('.'), images))
except FileNotFoundError:
    print("No such directory with the name {images}")
    exit()

if not os.path.exists(args.output):
    os.mkdir(args.output)

load_images = [cv.imread(img) for img in im]
im_batches = list(map(prep_image, load_images, [608 for x in range(len(im))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in load_images]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

leftover = 0
if (len(im_dim_list) % bs):
    leftover = 1

if bs != 1:
    num_batches = len(im) // bs + leftover
    im_batches = [torch.cat((im_batches[i* bs : min((i +  1)* bs, \
        len(im_batches))]))  for i in range(num_batches)] 

write = 0

for idx, batch in enumerate(im_batches):
    start = time.time()
    with torch.no_grad():
        pred = model(torch.Tensor(batch))

    pred = display(pred, confidence, num_classes, nms_threshold)
    end = time.time()
    if type(pred) == int:
        for im_num, image in enumerate(im[i* bs: min((idx +  1)* bs, len(im))]):
            im_id = idx * bs + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/ bs))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue
    
    pred[:,0] += idx * bs

    if not write:                     
        output = pred  
        write = 1
    else:
        output = torch.cat((output, pred))

    for im_num, image in enumerate(im[idx * bs: min((idx +  1)* bs, len(im))]):
        im_id = idx * bs + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        # print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/bs))
        # print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

# try:
#     output
# except NameError:
#     print ("No detections were made")
#     exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(608/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (608 - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (608 - scaling_factor*im_dim_list[:,1].view(-1,1))/2
output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

colors = pkl.load(open("pallete", "rb"))

def out(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv.rectangle(img, c1, c2,color, 1)
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv.rectangle(img, c1, c2,color, -1)
    cv.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

list(map(lambda x: out(x, load_images), output))

det_names = pd.Series(im).apply(lambda x: "{}/det_{}".format(args.output,x.split("/")[-1]))
print(det_names)
list(map(cv.imwrite, det_names, load_images))
