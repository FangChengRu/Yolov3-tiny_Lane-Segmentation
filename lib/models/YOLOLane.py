import torch
import torch.nn as nn
import math
import os
from lib.utils import initialize_weights
from lib.models.common_yolov3 import Conv, Concat, Detect
from torch.nn import Upsample, MaxPool2d, ZeroPad2d
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
import sys
sys.path.append(os.getcwd())


class MCnet(nn.Module):
    def __init__(self, block_cfg, export):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 13
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        self.export = export

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        detector = self.model[self.detector_index]  # detector
        if isinstance(detector, Detect):
            s = 256  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #    print(x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            detector.anchors /= detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(detector)
            self.stride = detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) \
                    else [x if j == -1 else cache[j] for j in block.from_]       # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:     # save driving area segment result
                if not self.export:
                    # print('Had sigmoid.')
                    m = nn.Sigmoid()
                    out.append(m(x))
                else:
                    print('No sigmoid.')
                    out.append(x)
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (416 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(export=False):
    yolov3lane = [
        [20, 31, 42],  # Det_out_idx, LL_Segout_idx
        [-1, Conv, [3, 16, 3, 1]],  # 0
        [-1, MaxPool2d, [2, 2, 0]],  # 1-P1/2
        [-1, Conv, [16, 32, 3, 1]],
        [-1, MaxPool2d, [2, 2, 0]],  # 3-P2/4
        [-1, Conv, [32, 64, 3, 1]],
        [-1, MaxPool2d, [2, 2, 0]],  # 5-P3/8
        [-1, Conv, [64, 128, 3, 1]],
        [-1, MaxPool2d, [2, 2, 0]],  # 7-P4/16
        [-1, Conv, [128, 256, 3, 1]],
        [-1, MaxPool2d, [2, 2, 0]],  # 9-P5/32
        [-1, Conv, [256, 512, 3, 1]],
        [-1, ZeroPad2d, [[0, 1, 0, 1]]],  # 11
        [-1, MaxPool2d, [2, 1, 0]],  # 12         #Encoder

        [-1, Conv, [512, 1024, 3, 1]],
        [-1, Conv, [1024, 256, 1, 1]],
        [-1, Conv, [256, 512, 3, 1]],  # 15 (P5/32-large)

        [-2, Conv, [256, 128, 1, 1]],
        [-1, Upsample, [None, 2, 'nearest']],
        [[-1, 8], Concat, [1]],  # cat backbone P4
        [-1, Conv, [384, 256, 3, 1]],  # 19 (P4/16-medium)

        [[19, 15], Detect, [13, [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]], [256, 512], export]],
        # 20, (class_num, anchor, channel)

        [12, Conv, [512, 256, 3, 1]],  # 21
        [-1, Upsample, [None, 2, 'nearest']],  # 22
        [-1, Conv, [256, 128, 3, 1]],  # 23
        [-1, Upsample, [None, 2, 'nearest']],  # 24
        [-1, Conv, [128, 64, 3, 1]],  # 25
        [-1, Upsample, [None, 2, 'nearest']],  # 26
        [-1, Conv, [64, 32, 3, 1]],  # 27
        [-1, Upsample, [None, 2, 'nearest']],  # 28
        [-1, Conv, [32, 16, 3, 1]],  # 29
        [-1, Upsample, [None, 2, 'nearest']],  # 30
        [-1, Conv, [16, 2, 3, 1]],  # 31 Driving area segmentation head

        [12, Conv, [512, 256, 3, 1]],  # 32
        [-1, Upsample, [None, 2, 'nearest']],  # 33
        [-1, Conv, [256, 128, 3, 1]],  # 34
        [-1, Upsample, [None, 2, 'nearest']],  # 35
        [-1, Conv, [128, 64, 3, 1]],  # 36
        [-1, Upsample, [None, 2, 'nearest']],  # 37
        [-1, Conv, [64, 32, 3, 1]],  # 38
        [-1, Upsample, [None, 2, 'nearest']],  # 39
        [-1, Conv, [32, 16, 3, 1]],  # 40
        [-1, Upsample, [None, 2, 'nearest']],  # 41
        [-1, Conv, [16, 2, 3, 1]]  # 42 Lane line segmentation head
    ]
    m_block_cfg = yolov3lane
    model_build = MCnet(m_block_cfg, export)
    return model_build


if __name__ == "__main__":
    model = get_net(export=False)
    input_ = torch.randn((1, 3, 416, 416))
    metric = SegmentationMetric(2)
    model_output = model(input_)
    detects_obj, dring_area_seg, lane_line_seg = model_output
    for det in detects_obj:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
