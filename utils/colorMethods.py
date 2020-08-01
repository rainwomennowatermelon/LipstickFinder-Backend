#!/usr/bin/python
# -*- encoding: utf-8 -*-

import math

def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))
    
def HexadecimalColor2RGB(HexadecimalColor):
    # convert HEX to base 10 for every digit, then calculate RGB
    digits = [int('0x' + d, 0) for d in HexadecimalColor.replace('#', '')]
    RGB = [digits[0] * 16 + digits[1], digits[2] * 16 + digits[3], digits[4] * 16 + digits[5]]
    return RGB

def hexaToRgb(hexa):
    r = int(hexa[1:3], 16)
    g = int(hexa[3:5], 16)
    b = int(hexa[5:7], 16)
    return [r,g,b]
