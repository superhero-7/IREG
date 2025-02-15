import re
import torch
import collections
import logging
import numpy as np
import torch.distributed as dist

def get_area(pos):
    """
    Args
        pos: [B, N, 4]
            (x1, x2, y1, y2)

    Return
        area : [B, N]
    """
    # [B, N]
    height = pos[:, :, 3] - pos[:, :, 2]
    width = pos[:, :, 1] - pos[:, :, 0]
    area = height * width
    return area

# 喵喵喵？没有搞懂是肾么东西
def get_relative_distance(pos):
    """
    Args
        pos: [B, N, 4]
            (x1, x2, y1, y2)

    Return
        out : [B, N, N, 4]
    """
    # B, N = pos.size()[:-1]

    # [B, N, N, 4]
    relative_distance = pos.unsqueeze(1) - pos.unsqueeze(2)
    return relative_distance


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)

# 这种好东西果然大家都用啊！
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    # pdb.set_trace()
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def get_iou(anchors, gt_boxes):
    """
    anchors: (N, 4) torch floattensor
    gt_boxes: (K, 4) torch floattensor
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)

    if gt_boxes.size() == (4,):
        gt_boxes = gt_boxes.view(1, 4)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
        (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) *
        (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    # 实现IOU的关键点就在这里，其实贼简单就写出来了...逻辑上一点点小技巧就OK
    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    # x,y是左下角， boxes[:, 0:2]是x,y， boxes[:, 0:2] + boxes[:, 2:4]是右上角坐标
    # 但不知道为什么要减1
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))
