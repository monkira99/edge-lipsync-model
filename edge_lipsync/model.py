#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hardcoded architecture generated from Emma/dh_model.param (NCNN layer count = 165).
CONV_DEFS: dict[str, dict[str, Any]] = {'convrelu_0': {'op': 'Convolution', 'params': {0: 16, 1: 3, 11: 3, 12: 1, 13: 1, 14: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 144, 9: 1}}, 'convrelu_1': {'op': 'Convolution', 'params': {0: 32, 1: 3, 11: 3, 12: 1, 13: 1, 14: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 4608, 9: 1}}, 'conv_2': {'op': 'Convolution', 'params': {0: 32, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 9216}}, 'convrelu_2': {'op': 'Convolution', 'params': {0: 64, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 18432, 9: 1}}, 'convrelu_3': {'op': 'Convolution', 'params': {0: 128, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 73728, 9: 1}}, 'convrelu_4': {'op': 'Convolution', 'params': {0: 128, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 147456, 9: 1}}, 'convrelu_5': {'op': 'Convolution', 'params': {0: 128, 1: 3, 11: 3, 12: 1, 13: 2, 14: 2, 2: 1, 3: 2, 4: 2, 5: 1, 6: 147456, 9: 1}}, 'conv_7': {'op': 'Convolution', 'params': {0: 128, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 147456}}, 'convrelu_6': {'op': 'Convolution', 'params': {0: 16, 1: 3, 11: 3, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 864, 7: 1, 9: 1}}, 'convrelu_7': {'op': 'Convolution', 'params': {0: 32, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 4608, 9: 1}}, 'convdwrelu_0': {'op': 'ConvolutionDepthWise', 'params': {0: 32, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 288, 7: 32, 9: 1}}, 'convrelu_8': {'op': 'Convolution', 'params': {0: 16, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 512, 9: 1}}, 'convrelu_9': {'op': 'Convolution', 'params': {0: 96, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 1536, 9: 1}}, 'convdwrelu_1': {'op': 'ConvolutionDepthWise', 'params': {0: 96, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 864, 7: 96, 9: 1}}, 'conv_11': {'op': 'Convolution', 'params': {0: 24, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 2304}}, 'convrelu_10': {'op': 'Convolution', 'params': {0: 144, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 3456, 9: 1}}, 'convdwrelu_2': {'op': 'ConvolutionDepthWise', 'params': {0: 144, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1296, 7: 144, 9: 1}}, 'conv_13': {'op': 'Convolution', 'params': {0: 24, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 3456}}, 'convrelu_11': {'op': 'Convolution', 'params': {0: 144, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 3456, 9: 1}}, 'convdwrelu_3': {'op': 'ConvolutionDepthWise', 'params': {0: 144, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1296, 7: 144, 9: 1}}, 'conv_15': {'op': 'Convolution', 'params': {0: 32, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 4608}}, 'convrelu_12': {'op': 'Convolution', 'params': {0: 192, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 6144, 9: 1}}, 'convdwrelu_4': {'op': 'ConvolutionDepthWise', 'params': {0: 192, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1728, 7: 192, 9: 1}}, 'conv_17': {'op': 'Convolution', 'params': {0: 32, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 6144}}, 'convrelu_13': {'op': 'Convolution', 'params': {0: 192, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 6144, 9: 1}}, 'convdwrelu_5': {'op': 'ConvolutionDepthWise', 'params': {0: 192, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1728, 7: 192, 9: 1}}, 'conv_19': {'op': 'Convolution', 'params': {0: 32, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 6144}}, 'convrelu_14': {'op': 'Convolution', 'params': {0: 192, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 6144, 9: 1}}, 'convdwrelu_6': {'op': 'ConvolutionDepthWise', 'params': {0: 192, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1728, 7: 192, 9: 1}}, 'conv_21': {'op': 'Convolution', 'params': {0: 64, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 12288}}, 'convrelu_15': {'op': 'Convolution', 'params': {0: 384, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 24576, 9: 1}}, 'convdwrelu_7': {'op': 'ConvolutionDepthWise', 'params': {0: 384, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3456, 7: 384, 9: 1}}, 'conv_23': {'op': 'Convolution', 'params': {0: 64, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 24576}}, 'convrelu_16': {'op': 'Convolution', 'params': {0: 384, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 24576, 9: 1}}, 'convdwrelu_8': {'op': 'ConvolutionDepthWise', 'params': {0: 384, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3456, 7: 384, 9: 1}}, 'conv_25': {'op': 'Convolution', 'params': {0: 64, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 24576}}, 'convrelu_17': {'op': 'Convolution', 'params': {0: 384, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 24576, 9: 1}}, 'convdwrelu_9': {'op': 'ConvolutionDepthWise', 'params': {0: 384, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3456, 7: 384, 9: 1}}, 'conv_27': {'op': 'Convolution', 'params': {0: 64, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 24576}}, 'convrelu_18': {'op': 'Convolution', 'params': {0: 384, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 24576, 9: 1}}, 'convdwrelu_10': {'op': 'ConvolutionDepthWise', 'params': {0: 384, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3456, 7: 384, 9: 1}}, 'conv_29': {'op': 'Convolution', 'params': {0: 96, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 36864}}, 'convrelu_19': {'op': 'Convolution', 'params': {0: 576, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 55296, 9: 1}}, 'convdwrelu_11': {'op': 'ConvolutionDepthWise', 'params': {0: 576, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 5184, 7: 576, 9: 1}}, 'conv_31': {'op': 'Convolution', 'params': {0: 96, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 55296}}, 'convrelu_20': {'op': 'Convolution', 'params': {0: 576, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 55296, 9: 1}}, 'convdwrelu_12': {'op': 'ConvolutionDepthWise', 'params': {0: 576, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 5184, 7: 576, 9: 1}}, 'conv_33': {'op': 'Convolution', 'params': {0: 96, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 55296}}, 'convrelu_21': {'op': 'Convolution', 'params': {0: 576, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 55296, 9: 1}}, 'convdwrelu_13': {'op': 'ConvolutionDepthWise', 'params': {0: 576, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 5184, 7: 576, 9: 1}}, 'conv_35': {'op': 'Convolution', 'params': {0: 160, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 92160}}, 'convrelu_22': {'op': 'Convolution', 'params': {0: 960, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 153600, 9: 1}}, 'convdwrelu_14': {'op': 'ConvolutionDepthWise', 'params': {0: 960, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 8640, 7: 960, 9: 1}}, 'conv_37': {'op': 'Convolution', 'params': {0: 160, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 153600}}, 'convrelu_23': {'op': 'Convolution', 'params': {0: 960, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 153600, 9: 1}}, 'convdwrelu_15': {'op': 'ConvolutionDepthWise', 'params': {0: 960, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 8640, 7: 960, 9: 1}}, 'conv_39': {'op': 'Convolution', 'params': {0: 160, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 153600}}, 'convrelu_24': {'op': 'Convolution', 'params': {0: 960, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 153600, 9: 1}}, 'convdwrelu_16': {'op': 'ConvolutionDepthWise', 'params': {0: 960, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 8640, 7: 960, 9: 1}}, 'conv_41': {'op': 'Convolution', 'params': {0: 320, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 307200}}, 'convrelu_25': {'op': 'Convolution', 'params': {0: 320, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 102400, 9: 1}}, 'deconvrelu_0': {'op': 'Deconvolution', 'params': {0: 320, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 18: 0, 19: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 40960, 9: 1}}, 'convrelu_26': {'op': 'Convolution', 'params': {0: 3840, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 2457600, 9: 1}}, 'convdwrelu_17': {'op': 'ConvolutionDepthWise', 'params': {0: 3840, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 34560, 7: 3840, 9: 1}}, 'conv_44': {'op': 'Convolution', 'params': {0: 320, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1228800}}, 'deconvrelu_1': {'op': 'Deconvolution', 'params': {0: 160, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 18: 1, 19: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 460800, 9: 1}}, 'convrelu_27': {'op': 'Convolution', 'params': {0: 1536, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 393216, 9: 1}}, 'convdwrelu_18': {'op': 'ConvolutionDepthWise', 'params': {0: 1536, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 13824, 7: 1536, 9: 1}}, 'conv_46': {'op': 'Convolution', 'params': {0: 128, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 196608}}, 'deconvrelu_2': {'op': 'Deconvolution', 'params': {0: 64, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 18: 1, 19: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 73728, 9: 1}}, 'convrelu_28': {'op': 'Convolution', 'params': {0: 576, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 55296, 9: 1}}, 'convdwrelu_19': {'op': 'ConvolutionDepthWise', 'params': {0: 576, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 5184, 7: 576, 9: 1}}, 'conv_48': {'op': 'Convolution', 'params': {0: 48, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 27648}}, 'deconvrelu_3': {'op': 'Deconvolution', 'params': {0: 24, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 18: 1, 19: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 10368, 9: 1}}, 'convrelu_29': {'op': 'Convolution', 'params': {0: 288, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 13824, 9: 1}}, 'convdwrelu_20': {'op': 'ConvolutionDepthWise', 'params': {0: 288, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2592, 7: 288, 9: 1}}, 'conv_50': {'op': 'Convolution', 'params': {0: 24, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 6912}}, 'deconvrelu_4': {'op': 'Deconvolution', 'params': {0: 16, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 18: 1, 19: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 3456, 9: 1}}, 'convrelu_30': {'op': 'Convolution', 'params': {0: 192, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 6144, 9: 1}}, 'convdwrelu_21': {'op': 'ConvolutionDepthWise', 'params': {0: 192, 1: 3, 11: 3, 12: 1, 13: 1, 14: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1728, 7: 192, 9: 1}}, 'conv_52': {'op': 'Convolution', 'params': {0: 16, 1: 1, 11: 1, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 3072}}, 'deconvrelu_5': {'op': 'Deconvolution', 'params': {0: 8, 1: 3, 11: 3, 12: 1, 13: 2, 14: 1, 18: 1, 19: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1152, 9: 1}}, 'conv_188': {'op': 'Convolution', 'params': {0: 3, 1: 3, 11: 3, 12: 1, 13: 1, 14: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 216, 7: 1}}}
GN_DEFS: dict[str, dict[int, Any]] = {'gn_59': {0: 1, 1: 24, 2: 1e-05, 3: 1}, 'gn_60': {0: 1, 1: 24, 2: 1e-05, 3: 1}, 'gn_61': {0: 1, 1: 32, 2: 1e-05, 3: 1}, 'gn_62': {0: 1, 1: 32, 2: 1e-05, 3: 1}, 'gn_63': {0: 1, 1: 32, 2: 1e-05, 3: 1}, 'gn_64': {0: 1, 1: 64, 2: 1e-05, 3: 1}, 'gn_65': {0: 1, 1: 64, 2: 1e-05, 3: 1}, 'gn_66': {0: 1, 1: 64, 2: 1e-05, 3: 1}, 'gn_67': {0: 1, 1: 64, 2: 1e-05, 3: 1}, 'gn_68': {0: 1, 1: 96, 2: 1e-05, 3: 1}, 'gn_69': {0: 1, 1: 96, 2: 1e-05, 3: 1}, 'gn_70': {0: 1, 1: 96, 2: 1e-05, 3: 1}, 'gn_71': {0: 1, 1: 160, 2: 1e-05, 3: 1}, 'gn_72': {0: 1, 1: 160, 2: 1e-05, 3: 1}, 'gn_73': {0: 1, 1: 160, 2: 1e-05, 3: 1}, 'gn_74': {0: 1, 1: 320, 2: 1e-05, 3: 1}, 'gn_75': {0: 1, 1: 320, 2: 1e-05, 3: 1}, 'gn_76': {0: 1, 1: 128, 2: 1e-05, 3: 1}, 'gn_77': {0: 1, 1: 48, 2: 1e-05, 3: 1}, 'gn_78': {0: 1, 1: 24, 2: 1e-05, 3: 1}, 'gn_79': {0: 1, 1: 16, 2: 1e-05, 3: 1}}
PAD_DEFS: dict[str, dict[int, Any]] = {'pad_185': {0: 1, 1: 1, 2: 1, 3: 1, 4: 2}, 'pad_187': {0: 1, 1: 1, 2: 1, 3: 1, 4: 2}}
WEIGHT_ORDER: list[tuple[str, str]] = [('Convolution', 'convrelu_0'), ('Convolution', 'convrelu_1'), ('Convolution', 'conv_2'), ('Convolution', 'convrelu_2'), ('Convolution', 'convrelu_3'), ('Convolution', 'convrelu_4'), ('Convolution', 'convrelu_5'), ('Convolution', 'conv_7'), ('Convolution', 'convrelu_6'), ('Convolution', 'convrelu_7'), ('ConvolutionDepthWise', 'convdwrelu_0'), ('Convolution', 'convrelu_8'), ('Convolution', 'convrelu_9'), ('ConvolutionDepthWise', 'convdwrelu_1'), ('Convolution', 'conv_11'), ('GroupNorm', 'gn_59'), ('Convolution', 'convrelu_10'), ('ConvolutionDepthWise', 'convdwrelu_2'), ('Convolution', 'conv_13'), ('GroupNorm', 'gn_60'), ('Convolution', 'convrelu_11'), ('ConvolutionDepthWise', 'convdwrelu_3'), ('Convolution', 'conv_15'), ('GroupNorm', 'gn_61'), ('Convolution', 'convrelu_12'), ('ConvolutionDepthWise', 'convdwrelu_4'), ('Convolution', 'conv_17'), ('GroupNorm', 'gn_62'), ('Convolution', 'convrelu_13'), ('ConvolutionDepthWise', 'convdwrelu_5'), ('Convolution', 'conv_19'), ('GroupNorm', 'gn_63'), ('Convolution', 'convrelu_14'), ('ConvolutionDepthWise', 'convdwrelu_6'), ('Convolution', 'conv_21'), ('GroupNorm', 'gn_64'), ('Convolution', 'convrelu_15'), ('ConvolutionDepthWise', 'convdwrelu_7'), ('Convolution', 'conv_23'), ('GroupNorm', 'gn_65'), ('Convolution', 'convrelu_16'), ('ConvolutionDepthWise', 'convdwrelu_8'), ('Convolution', 'conv_25'), ('GroupNorm', 'gn_66'), ('Convolution', 'convrelu_17'), ('ConvolutionDepthWise', 'convdwrelu_9'), ('Convolution', 'conv_27'), ('GroupNorm', 'gn_67'), ('Convolution', 'convrelu_18'), ('ConvolutionDepthWise', 'convdwrelu_10'), ('Convolution', 'conv_29'), ('GroupNorm', 'gn_68'), ('Convolution', 'convrelu_19'), ('ConvolutionDepthWise', 'convdwrelu_11'), ('Convolution', 'conv_31'), ('GroupNorm', 'gn_69'), ('Convolution', 'convrelu_20'), ('ConvolutionDepthWise', 'convdwrelu_12'), ('Convolution', 'conv_33'), ('GroupNorm', 'gn_70'), ('Convolution', 'convrelu_21'), ('ConvolutionDepthWise', 'convdwrelu_13'), ('Convolution', 'conv_35'), ('GroupNorm', 'gn_71'), ('Convolution', 'convrelu_22'), ('ConvolutionDepthWise', 'convdwrelu_14'), ('Convolution', 'conv_37'), ('GroupNorm', 'gn_72'), ('Convolution', 'convrelu_23'), ('ConvolutionDepthWise', 'convdwrelu_15'), ('Convolution', 'conv_39'), ('GroupNorm', 'gn_73'), ('Convolution', 'convrelu_24'), ('ConvolutionDepthWise', 'convdwrelu_16'), ('Convolution', 'conv_41'), ('GroupNorm', 'gn_74'), ('Convolution', 'convrelu_25'), ('Deconvolution', 'deconvrelu_0'), ('Convolution', 'convrelu_26'), ('ConvolutionDepthWise', 'convdwrelu_17'), ('Convolution', 'conv_44'), ('GroupNorm', 'gn_75'), ('Deconvolution', 'deconvrelu_1'), ('Convolution', 'convrelu_27'), ('ConvolutionDepthWise', 'convdwrelu_18'), ('Convolution', 'conv_46'), ('GroupNorm', 'gn_76'), ('Deconvolution', 'deconvrelu_2'), ('Convolution', 'convrelu_28'), ('ConvolutionDepthWise', 'convdwrelu_19'), ('Convolution', 'conv_48'), ('GroupNorm', 'gn_77'), ('Deconvolution', 'deconvrelu_3'), ('Convolution', 'convrelu_29'), ('ConvolutionDepthWise', 'convdwrelu_20'), ('Convolution', 'conv_50'), ('GroupNorm', 'gn_78'), ('Deconvolution', 'deconvrelu_4'), ('Convolution', 'convrelu_30'), ('ConvolutionDepthWise', 'convdwrelu_21'), ('Convolution', 'conv_52'), ('GroupNorm', 'gn_79'), ('Deconvolution', 'deconvrelu_5'), ('Convolution', 'conv_188')]


class _NcnnBinReader:
    FP16_FLAG = 0x01306B47
    FP32_FLAG = 0x00000000

    def __init__(self, bin_path: str | Path) -> None:
        self.data = Path(bin_path).read_bytes()
        self.offset = 0

    def _need(self, nbytes: int) -> None:
        if self.offset + nbytes > len(self.data):
            raise ValueError('NCNN bin truncated')

    def _read_u32(self) -> int:
        self._need(4)
        v = int.from_bytes(self.data[self.offset:self.offset+4], 'little')
        self.offset += 4
        return v

    def read_blob(self, numel: int) -> torch.Tensor:
        if numel <= 0:
            return torch.empty(0, dtype=torch.float32)
        flag = self._read_u32()
        if flag == self.FP16_FLAG:
            nbytes = numel * 2
            self._need(nbytes)
            mv = memoryview(self.data)[self.offset:self.offset+nbytes]
            self.offset += nbytes
            return torch.frombuffer(bytearray(mv), dtype=torch.float16).to(torch.float32)
        if flag == self.FP32_FLAG:
            nbytes = numel * 4
            self._need(nbytes)
            mv = memoryview(self.data)[self.offset:self.offset+nbytes]
            self.offset += nbytes
            return torch.frombuffer(bytearray(mv), dtype=torch.float32)
        # legacy blob format without flag
        self.offset -= 4
        nbytes = numel * 4
        self._need(nbytes)
        mv = memoryview(self.data)[self.offset:self.offset+nbytes]
        self.offset += nbytes
        return torch.frombuffer(bytearray(mv), dtype=torch.float32)

    def remaining(self) -> int:
        return len(self.data) - self.offset



class ConvBlock(nn.Module):
    """Single NCNN convolution/deconvolution op with optional ReLU."""

    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        p = cfg['params']
        out_ch = int(p.get(0, 0))
        kw = int(p.get(1, 1))
        kh = int(p.get(11, kw))
        dw = int(p.get(2, 1))
        dh = int(p.get(12, dw))
        sw = int(p.get(3, 1))
        sh = int(p.get(13, sw))
        pw = int(p.get(4, 0))
        ph = int(p.get(14, pw))
        groups = int(p.get(7, 1))
        bias = bool(int(p.get(5, 0)))

        if cfg['op'] == 'Deconvolution':
            opw = int(p.get(18, 0))
            oph = int(p.get(19, 0))
            self.op: nn.Module = nn.LazyConvTranspose2d(
                out_channels=out_ch,
                kernel_size=(kh, kw),
                stride=(sh, sw),
                padding=(ph, pw),
                output_padding=(oph, opw),
                dilation=(dh, dw),
                groups=groups,
                bias=bias,
            )
        else:
            self.op = nn.LazyConv2d(
                out_channels=out_ch,
                kernel_size=(kh, kw),
                stride=(sh, sw),
                padding=(ph, pw),
                dilation=(dh, dw),
                groups=groups,
                bias=bias,
            )

        self.use_relu = int(p.get(9, 0)) == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.op(x)
        if self.use_relu:
            y = F.relu(y, inplace=False)
        return y


class MBBlock(nn.Module):
    """MobileNet-style block: expand -> depthwise -> project -> GN -> ReLU (+ residual)."""

    def __init__(
        self,
        *,
        expand: ConvBlock,
        depthwise: ConvBlock,
        project: ConvBlock,
        gn: nn.GroupNorm,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.expand = expand
        self.depthwise = depthwise
        self.project = project
        self.gn = gn
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.depthwise(y)
        y = self.project(y)
        y = self.gn(y)
        y = F.relu(y, inplace=False)
        if self.use_residual:
            y = x + y
        return y


class UpBlock(nn.Module):
    """Decoder block: upsample -> concat skip -> MBBlock."""

    def __init__(self, *, up: ConvBlock, fuse: MBBlock) -> None:
        super().__init__()
        self.up = up
        self.fuse = fuse

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        y = self.up(x)
        if skip is not None:
            y = torch.cat([skip, y], dim=1)
        y = self.fuse(y)
        return y


class DuixUNet(nn.Module):
    """
    Hardcoded encoder-decoder Duix UNet (blockified version).

    Inputs:
      - face:  [B, 6, H, W]
      - audio: [B, 20, 256] or [B, 1, 20, 256]
    Output:
      - [B, 3, H, W]
    """

    def __init__(self) -> None:
        super().__init__()
        self._name_to_module: dict[str, nn.Module] = {}
        self._name_to_prefix: dict[str, str] = {}
        self._build_blocks()

    def _build_conv(self, ncnn_name: str, prefix: str) -> ConvBlock:
        if ncnn_name in self._name_to_module:
            raise ValueError(f'Duplicate NCNN layer registration: {ncnn_name}')
        block = ConvBlock(ncnn_name, CONV_DEFS[ncnn_name])
        self._name_to_module[ncnn_name] = block.op
        self._name_to_prefix[ncnn_name] = f'{prefix}.op'
        return block

    def _build_gn(self, ncnn_name: str, prefix: str) -> nn.GroupNorm:
        if ncnn_name in self._name_to_module:
            raise ValueError(f'Duplicate NCNN layer registration: {ncnn_name}')
        p = GN_DEFS[ncnn_name]
        groups = int(p.get(0, 1))
        channels = int(p.get(1, 1))
        eps = float(p.get(2, 1e-5))
        affine = bool(int(p.get(3, 1)))
        gn = nn.GroupNorm(groups, channels, eps=eps, affine=affine)
        self._name_to_module[ncnn_name] = gn
        self._name_to_prefix[ncnn_name] = prefix
        return gn

    def _build_blocks(self) -> None:
        # Audio encoder
        self.a_conv0 = self._build_conv('convrelu_0', 'a_conv0')
        self.a_conv1 = self._build_conv('convrelu_1', 'a_conv1')
        self.a_res = self._build_conv('conv_2', 'a_res')
        self.a_conv2 = self._build_conv('convrelu_2', 'a_conv2')
        self.a_conv3 = self._build_conv('convrelu_3', 'a_conv3')
        self.a_conv4 = self._build_conv('convrelu_4', 'a_conv4')
        self.a_conv5 = self._build_conv('convrelu_5', 'a_conv5')
        self.a_out = self._build_conv('conv_7', 'a_out')

        # Face encoder stem
        self.f_stem0 = self._build_conv('convrelu_6', 'f_stem0')
        self.f_stem1 = self._build_conv('convrelu_7', 'f_stem1')
        self.f_stem2 = self._build_conv('convdwrelu_0', 'f_stem2')
        self.f_stem3 = self._build_conv('convrelu_8', 'f_stem3')

        # Face encoder MB blocks
        self.enc_mb1 = MBBlock(
            expand=self._build_conv('convrelu_9', 'enc_mb1.expand'),
            depthwise=self._build_conv('convdwrelu_1', 'enc_mb1.depthwise'),
            project=self._build_conv('conv_11', 'enc_mb1.project'),
            gn=self._build_gn('gn_59', 'enc_mb1.gn'),
            use_residual=False,
        )
        self.enc_mb2 = MBBlock(
            expand=self._build_conv('convrelu_10', 'enc_mb2.expand'),
            depthwise=self._build_conv('convdwrelu_2', 'enc_mb2.depthwise'),
            project=self._build_conv('conv_13', 'enc_mb2.project'),
            gn=self._build_gn('gn_60', 'enc_mb2.gn'),
            use_residual=True,
        )
        self.enc_mb3 = MBBlock(
            expand=self._build_conv('convrelu_11', 'enc_mb3.expand'),
            depthwise=self._build_conv('convdwrelu_3', 'enc_mb3.depthwise'),
            project=self._build_conv('conv_15', 'enc_mb3.project'),
            gn=self._build_gn('gn_61', 'enc_mb3.gn'),
            use_residual=False,
        )
        self.enc_mb4 = MBBlock(
            expand=self._build_conv('convrelu_12', 'enc_mb4.expand'),
            depthwise=self._build_conv('convdwrelu_4', 'enc_mb4.depthwise'),
            project=self._build_conv('conv_17', 'enc_mb4.project'),
            gn=self._build_gn('gn_62', 'enc_mb4.gn'),
            use_residual=True,
        )
        self.enc_mb5 = MBBlock(
            expand=self._build_conv('convrelu_13', 'enc_mb5.expand'),
            depthwise=self._build_conv('convdwrelu_5', 'enc_mb5.depthwise'),
            project=self._build_conv('conv_19', 'enc_mb5.project'),
            gn=self._build_gn('gn_63', 'enc_mb5.gn'),
            use_residual=True,
        )
        self.enc_mb6 = MBBlock(
            expand=self._build_conv('convrelu_14', 'enc_mb6.expand'),
            depthwise=self._build_conv('convdwrelu_6', 'enc_mb6.depthwise'),
            project=self._build_conv('conv_21', 'enc_mb6.project'),
            gn=self._build_gn('gn_64', 'enc_mb6.gn'),
            use_residual=False,
        )
        self.enc_mb7 = MBBlock(
            expand=self._build_conv('convrelu_15', 'enc_mb7.expand'),
            depthwise=self._build_conv('convdwrelu_7', 'enc_mb7.depthwise'),
            project=self._build_conv('conv_23', 'enc_mb7.project'),
            gn=self._build_gn('gn_65', 'enc_mb7.gn'),
            use_residual=True,
        )
        self.enc_mb8 = MBBlock(
            expand=self._build_conv('convrelu_16', 'enc_mb8.expand'),
            depthwise=self._build_conv('convdwrelu_8', 'enc_mb8.depthwise'),
            project=self._build_conv('conv_25', 'enc_mb8.project'),
            gn=self._build_gn('gn_66', 'enc_mb8.gn'),
            use_residual=True,
        )
        self.enc_mb9 = MBBlock(
            expand=self._build_conv('convrelu_17', 'enc_mb9.expand'),
            depthwise=self._build_conv('convdwrelu_9', 'enc_mb9.depthwise'),
            project=self._build_conv('conv_27', 'enc_mb9.project'),
            gn=self._build_gn('gn_67', 'enc_mb9.gn'),
            use_residual=True,
        )
        self.enc_mb10 = MBBlock(
            expand=self._build_conv('convrelu_18', 'enc_mb10.expand'),
            depthwise=self._build_conv('convdwrelu_10', 'enc_mb10.depthwise'),
            project=self._build_conv('conv_29', 'enc_mb10.project'),
            gn=self._build_gn('gn_68', 'enc_mb10.gn'),
            use_residual=False,
        )
        self.enc_mb11 = MBBlock(
            expand=self._build_conv('convrelu_19', 'enc_mb11.expand'),
            depthwise=self._build_conv('convdwrelu_11', 'enc_mb11.depthwise'),
            project=self._build_conv('conv_31', 'enc_mb11.project'),
            gn=self._build_gn('gn_69', 'enc_mb11.gn'),
            use_residual=True,
        )
        self.enc_mb12 = MBBlock(
            expand=self._build_conv('convrelu_20', 'enc_mb12.expand'),
            depthwise=self._build_conv('convdwrelu_12', 'enc_mb12.depthwise'),
            project=self._build_conv('conv_33', 'enc_mb12.project'),
            gn=self._build_gn('gn_70', 'enc_mb12.gn'),
            use_residual=True,
        )
        self.enc_mb13 = MBBlock(
            expand=self._build_conv('convrelu_21', 'enc_mb13.expand'),
            depthwise=self._build_conv('convdwrelu_13', 'enc_mb13.depthwise'),
            project=self._build_conv('conv_35', 'enc_mb13.project'),
            gn=self._build_gn('gn_71', 'enc_mb13.gn'),
            use_residual=False,
        )
        self.enc_mb14 = MBBlock(
            expand=self._build_conv('convrelu_22', 'enc_mb14.expand'),
            depthwise=self._build_conv('convdwrelu_14', 'enc_mb14.depthwise'),
            project=self._build_conv('conv_37', 'enc_mb14.project'),
            gn=self._build_gn('gn_72', 'enc_mb14.gn'),
            use_residual=True,
        )
        self.enc_mb15 = MBBlock(
            expand=self._build_conv('convrelu_23', 'enc_mb15.expand'),
            depthwise=self._build_conv('convdwrelu_15', 'enc_mb15.depthwise'),
            project=self._build_conv('conv_39', 'enc_mb15.project'),
            gn=self._build_gn('gn_73', 'enc_mb15.gn'),
            use_residual=True,
        )
        self.enc_mb16 = MBBlock(
            expand=self._build_conv('convrelu_24', 'enc_mb16.expand'),
            depthwise=self._build_conv('convdwrelu_16', 'enc_mb16.depthwise'),
            project=self._build_conv('conv_41', 'enc_mb16.project'),
            gn=self._build_gn('gn_74', 'enc_mb16.gn'),
            use_residual=False,
        )
        self.bottleneck_conv = self._build_conv('convrelu_25', 'bottleneck_conv')

        # Decoder
        self.dec_up0 = UpBlock(
            up=self._build_conv('deconvrelu_0', 'dec_up0.up'),
            fuse=MBBlock(
                expand=self._build_conv('convrelu_26', 'dec_up0.fuse.expand'),
                depthwise=self._build_conv('convdwrelu_17', 'dec_up0.fuse.depthwise'),
                project=self._build_conv('conv_44', 'dec_up0.fuse.project'),
                gn=self._build_gn('gn_75', 'dec_up0.fuse.gn'),
                use_residual=False,
            ),
        )
        self.dec_up1 = UpBlock(
            up=self._build_conv('deconvrelu_1', 'dec_up1.up'),
            fuse=MBBlock(
                expand=self._build_conv('convrelu_27', 'dec_up1.fuse.expand'),
                depthwise=self._build_conv('convdwrelu_18', 'dec_up1.fuse.depthwise'),
                project=self._build_conv('conv_46', 'dec_up1.fuse.project'),
                gn=self._build_gn('gn_76', 'dec_up1.fuse.gn'),
                use_residual=False,
            ),
        )
        self.dec_up2 = UpBlock(
            up=self._build_conv('deconvrelu_2', 'dec_up2.up'),
            fuse=MBBlock(
                expand=self._build_conv('convrelu_28', 'dec_up2.fuse.expand'),
                depthwise=self._build_conv('convdwrelu_19', 'dec_up2.fuse.depthwise'),
                project=self._build_conv('conv_48', 'dec_up2.fuse.project'),
                gn=self._build_gn('gn_77', 'dec_up2.fuse.gn'),
                use_residual=False,
            ),
        )
        self.dec_up3 = UpBlock(
            up=self._build_conv('deconvrelu_3', 'dec_up3.up'),
            fuse=MBBlock(
                expand=self._build_conv('convrelu_29', 'dec_up3.fuse.expand'),
                depthwise=self._build_conv('convdwrelu_20', 'dec_up3.fuse.depthwise'),
                project=self._build_conv('conv_50', 'dec_up3.fuse.project'),
                gn=self._build_gn('gn_78', 'dec_up3.fuse.gn'),
                use_residual=False,
            ),
        )
        self.dec_up4 = UpBlock(
            up=self._build_conv('deconvrelu_4', 'dec_up4.up'),
            fuse=MBBlock(
                expand=self._build_conv('convrelu_30', 'dec_up4.fuse.expand'),
                depthwise=self._build_conv('convdwrelu_21', 'dec_up4.fuse.depthwise'),
                project=self._build_conv('conv_52', 'dec_up4.fuse.project'),
                gn=self._build_gn('gn_79', 'dec_up4.fuse.gn'),
                use_residual=False,
            ),
        )
        self.dec_final_up = self._build_conv('deconvrelu_5', 'dec_final_up')
        self.out_conv = self._build_conv('conv_188', 'out_conv')

    @staticmethod
    def _normalize_audio(audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            if audio.shape[1:] == (20, 256):
                return audio.unsqueeze(1)
            if audio.shape[1:] == (256, 20):
                return audio.transpose(1, 2).unsqueeze(1)
        if audio.dim() == 4 and audio.shape[1:] == (1, 20, 256):
            return audio
        raise ValueError(f'Unsupported audio shape {tuple(audio.shape)}')

    def _pad(self, name: str, x: torch.Tensor) -> torch.Tensor:
        p = PAD_DEFS[name]
        pad_top = int(p.get(0, 0))
        pad_bottom = int(p.get(1, 0))
        pad_left = int(p.get(2, 0))
        pad_right = int(p.get(3, 0))
        pad_type = int(p.get(4, 0))
        pad_val = float(p.get(5, 0.0))
        if pad_type == 0:
            return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_val)
        if pad_type == 1:
            return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        if pad_type == 2:
            return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        raise NotImplementedError(f'Unsupported pad type {pad_type} for {name}')

    def _remap_legacy_layers_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        remapped: dict[str, torch.Tensor] = {}

        for key, val in state_dict.items():
            layer_key = None
            if key.startswith('layers.'):
                layer_key = key[len('layers.'):]
            elif key.startswith('graph.layers.'):
                layer_key = key[len('graph.layers.'):]

            if layer_key is None:
                remapped[key] = val
                continue

            parts = layer_key.split('.', 1)
            if len(parts) != 2:
                raise KeyError(f'Unexpected legacy key format: {key}')
            layer_name, suffix = parts
            prefix = self._name_to_prefix.get(layer_name)
            if prefix is None:
                raise KeyError(f'Unknown legacy layer name: {layer_name}')
            remapped[f'{prefix}.{suffix}'] = val

        return remapped

    def forward(self, face: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if face.dim() != 4 or face.shape[1] != 6:
            raise ValueError(f'Face must be [B,6,H,W], got {tuple(face.shape)}')
        audio = self._normalize_audio(audio)

        # Audio encoder branch
        a = self.a_conv0(audio)
        a = self.a_conv1(a)
        a_res = a
        a = self.a_res(a)
        a = F.relu(a + a_res, inplace=False)
        a = self.a_conv2(a)
        a = self.a_conv3(a)
        a = self.a_conv4(a)
        a = self.a_conv5(a)
        a_res2 = a
        a = self.a_out(a)
        a_feat = F.relu(a + a_res2, inplace=False)

        # Face encoder stem
        x = self._pad('pad_185', face)
        x = self.f_stem0(x)
        x = self.f_stem1(x)
        x = self.f_stem2(x)
        x = self.f_stem3(x)
        skip23 = x

        # Face encoder
        x = self.enc_mb1(x)
        x = self.enc_mb2(x)
        skip38 = x

        x = self.enc_mb3(x)
        x = self.enc_mb4(x)
        x = self.enc_mb5(x)
        skip61 = x

        x = self.enc_mb6(x)
        x = self.enc_mb7(x)
        x = self.enc_mb8(x)
        x = self.enc_mb9(x)

        x = self.enc_mb10(x)
        x = self.enc_mb11(x)
        x = self.enc_mb12(x)
        skip113 = x

        x = self.enc_mb13(x)
        x = self.enc_mb14(x)
        x = self.enc_mb15(x)
        x = self.enc_mb16(x)
        bottleneck = self.bottleneck_conv(x)

        # Decoder
        x = self.dec_up0(a_feat, bottleneck)
        x = self.dec_up1(x, skip113)
        x = self.dec_up2(x, skip61)
        x = self.dec_up3(x, skip38)
        x = self.dec_up4(x, skip23)

        x = self.dec_final_up(x)
        x = self._pad('pad_187', x)
        x = self.out_conv(x)
        out = torch.tanh(x)
        return out

    def load_ncnn_bin(self, bin_path: str | Path, face_size: int = 160, device: str = 'cpu') -> dict[str, int]:
        self.to(device).eval()
        with torch.no_grad():
            _ = self(
                torch.zeros(1, 6, face_size, face_size, device=device),
                torch.zeros(1, 20, 256, device=device),
            )

        reader = _NcnnBinReader(bin_path)
        loaded_blobs = 0
        loaded_params = 0

        with torch.no_grad():
            for op, name in WEIGHT_ORDER:
                mod = self._name_to_module[name]
                if op in ('Convolution', 'ConvolutionDepthWise', 'Deconvolution'):
                    if not hasattr(mod, 'weight'):
                        raise TypeError(f'Layer {name} has no weight tensor')
                    weight = mod.weight  # type: ignore[attr-defined]
                    w_flat = reader.read_blob(weight.numel())
                    if op == 'Deconvolution':
                        in_ch = mod.in_channels  # type: ignore[attr-defined]
                        out_ch = mod.out_channels  # type: ignore[attr-defined]
                        groups = mod.groups  # type: ignore[attr-defined]
                        kh, kw = mod.kernel_size  # type: ignore[attr-defined]
                        in_g = in_ch // groups
                        out_g = out_ch // groups
                        w = w_flat.view(groups, out_g, in_g, kh, kw).permute(0, 2, 1, 3, 4).contiguous()
                        w = w.view_as(weight)
                    else:
                        w = w_flat.view_as(weight)
                    weight.copy_(w.to(device=device, dtype=weight.dtype))
                    loaded_blobs += 1
                    loaded_params += weight.numel()

                    bias = getattr(mod, 'bias', None)
                    if bias is not None:
                        b = reader.read_blob(bias.numel()).view_as(bias)
                        bias.copy_(b.to(device=device, dtype=bias.dtype))
                        loaded_blobs += 1
                        loaded_params += bias.numel()
                elif op == 'GroupNorm':
                    if not isinstance(mod, nn.GroupNorm):
                        raise TypeError(f'Expected GroupNorm for {name}, got {type(mod)}')
                    if mod.affine:
                        w = reader.read_blob(mod.weight.numel()).view_as(mod.weight)
                        b = reader.read_blob(mod.bias.numel()).view_as(mod.bias)
                        mod.weight.copy_(w.to(device=device, dtype=mod.weight.dtype))
                        mod.bias.copy_(b.to(device=device, dtype=mod.bias.dtype))
                        loaded_blobs += 2
                        loaded_params += mod.weight.numel() + mod.bias.numel()

        return {
            'loaded_blobs': loaded_blobs,
            'loaded_params': loaded_params,
            'remaining_bytes': reader.remaining(),
        }


def save_ckpt(model: DuixUNet, ckpt_path: str | Path, face_size: int = 160, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        'format': 'duix_unet_hardcoded_blocks_ckpt_v2',
        'state_dict': model.state_dict(),
        'face_size': face_size,
        'extra': extra or {},
        'spec_count': 165,
    }
    out = Path(ckpt_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    try:
        torch.save(payload, str(tmp))
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_ckpt(ckpt_path: str | Path, *, map_location: str | torch.device = 'cpu', strict: bool = True) -> DuixUNet:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise ValueError(f'Invalid ckpt format: {ckpt_path}')

    model = DuixUNet()
    state_dict = ckpt['state_dict']
    if not isinstance(state_dict, dict):
        raise ValueError(f'Invalid state_dict in ckpt: {ckpt_path}')

    if any(k.startswith('layers.') or k.startswith('graph.layers.') for k in state_dict.keys()):
        state_dict = model._remap_legacy_layers_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def _demo(ckpt_path: str, device: str) -> None:
    model = load_ckpt(ckpt_path, map_location=device).to(device).eval()
    face = torch.randn(1, 6, 160, 160, device=device)
    audio = torch.randn(1, 20, 256, device=device)
    with torch.no_grad():
        out = model(face, audio)
    print('loaded:', ckpt_path)
    print('out shape:', tuple(out.shape))
    print('out dtype:', out.dtype)
    print('out range:', float(out.min()), float(out.max()))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load hardcoded-blocks Duix ckpt and run a forward sanity check.')
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    _demo(args.ckpt, args.device)
