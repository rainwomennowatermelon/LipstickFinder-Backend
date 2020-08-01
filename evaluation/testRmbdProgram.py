#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.model import BiSeNet

import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import cv2
import math
import time
import json
import io
import attr
import os

# Convert the HSV color space to the RGB color space
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

# Convert the RGB color space to the HSV color space
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx
    # h:0-360, s:0-1, v:0-1
    # H:0-180, S:0-255, V:0-255
    H = h / 2
    S = s * 255.0
    V = v * 255.0
    return H, S, V

# Color distance between two RGB colors
def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

# RMBD
def rmDarkBright(imgArray, parsing):
    for i in range(512):
        for j in range(512):
            rgb = imgArray[i][j]
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
            h, s, v = rgb2hsv(r, g, b)
            if (parsing[i][j] == 12 or parsing[i][j] == 13) and (v < 5 or v > 250):
                parsing[i][j] = 1
    return parsing

# Get ret_dic
with open('./json/lipsticksMod.json', 'r', encoding='utf-8') as f:
    ret_dic = json.load(f)
    sum = len(ret_dic)
    RGB_array = np.zeros((sum, 3), dtype=int)
    for i in range(sum):
        color_value = ret_dic[i]['color']
        ret_dic[i]['distance'] = 999
        ret_dic[i]['rgb'] = [0, 0, 0]
        ret_dic[i]['rgb'][0] = int(color_value[1:3], 16)
        ret_dic[i]['rgb'][1] = int(color_value[3:5], 16)
        ret_dic[i]['rgb'][2] = int(color_value[5:7], 16)

# Load into memory
# Load model
n_classes = 19
net = BiSeNet(n_classes=n_classes)
save_pth = '../res/cp/79999_iter.pth'
net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
net.eval()

# Tensor
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Count correct number and fail number
shootCount = 0
failCount = 0

# Test sub-dataset dir
testDirPath = "./testDataset/dior/star" # Change para


for img in os.listdir(testDirPath):
    with torch.no_grad():
        image = Image.open(testDirPath + "/" + img).convert('RGB')
        imgArray = np.array(image)
        image = to_tensor(image)
        image = torch.unsqueeze(image, 0)
        out = net(image)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        parsing = rmDarkBright(imgArray, parsing)
        upperLipPos = np.where(parsing == 12)
        lowerLipPos = np.where(parsing == 13)

        pointsCount = len(upperLipPos[0]) + len(lowerLipPos[0])
        pointsCountUpper = len(upperLipPos[0])
        pointsCountLower = len(lowerLipPos[0])
        rgb = [0, 0, 0]
        if len(upperLipPos) > 0:
            rgbUpper = np.sum(imgArray[upperLipPos[0], upperLipPos[1], :], axis=0)
            rgb = np.sum([rgb, rgbUpper], axis=0)
        if len(lowerLipPos) > 0:
            rgbLower = np.sum(imgArray[lowerLipPos[0], lowerLipPos[1], :], axis=0)
            rgb = np.sum([rgb, rgbLower], axis=0)
        if pointsCount > 0:
            res = [math.floor(i / pointsCount) for i in rgb]
            for i in range(sum):
                ret_dic[i]['distance'] = ColourDistance(res, ret_dic[i]['rgb'])
            predictTmp = sorted(ret_dic, key=lambda x: float(x['distance']))[:20]
            predictRes = []
            for lipstick in predictTmp:
                predictRes.append(lipstick['id'])

            front = img.split('.')[0]
            imgId = front.split('_')[1]
            print('imgId: ' + imgId + ', extractRGB: ' + str(res) + ', predictRes: ' + str(predictRes) + ', shoot: ' + str(int(imgId) in predictRes))
            if int(imgId) in predictRes:
                shootCount += 1
            else:
                failCount += 1
print(testDirPath + ': ' + str(shootCount) + ' success, ' + str(failCount) + ' fails, rate: ' + str(shootCount/(shootCount+failCount)))
