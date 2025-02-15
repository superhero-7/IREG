# coding=utf-8

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from detectron2_given_box_maxnms import extract, DIM

from pycocotools.coco import COCO


class RefCOCODataset(Dataset):
    def __init__(self, refcoco_dir, refcoco_images_dir, coco_dir, split='val'):

        self.image_dir = refcoco_images_dir

        # coco_train_annFile = coco_dir.joinpath('annotations/instances_train2014.json')
        # self.coco = COCO(coco_train_annFile)

        assert split in ['train', 'val', 'test']

        workspace_dir = Path(__file__).resolve().parent.parent
        refcoco_util_dir = workspace_dir.joinpath('VL-T5', 'src')
        import sys
        sys.path.append(str(refcoco_util_dir))
        from eval_utils.refcoco_utils import REFER
        self.refer = REFER('refcocog', 'umd')

        ref_ids = self.refer.getRefIds(split=split)

        id2dets = {}
        img_ids = []
        image_fns = []
        for ref_id in ref_ids:
            ref = self.refer.Refs[ref_id]
            print(ref)
            img_id = ref['image_id']

            if img_id not in img_ids:
                img_ids.append(img_id)

                fn_ann = ref['file_name']

                # COCO_train2014_000000419645_398406.jpg
                # COCO_train2014_000000419645.jpg

                suffix = fn_ann.split('.')[-1]

                fname = '_'.join(fn_ann.split('_')[:-1]) + '.' + suffix

                image_fns.append(fname)

                detections = self.refer.imgToAnns[img_id]

                id2dets[img_id] = detections

        self.image_ids = img_ids
        self.image_fns = image_fns
        self.id2dets = id2dets

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]
        image_fn = self.image_fns[idx]
        image_path = self.image_dir.joinpath(image_fn)

        assert Path(image_path).exists(), image_path

        img = cv2.imread(str(image_path))

        H, W, C = img.shape

        dets = self.id2dets[image_id]
        # cat_names = [det['category_name'] for det in dets]

        boxes = []
        for i, region in enumerate([det['bbox'] for det in dets]):
            # (x1, y1, x2, y2)
            x, y, w, h = region[:4]
            x1, y1, x2, y2 = x, y, x+w, y+h

            # x1, y1, x2, y2 = region[:4]

            assert x2 <= W, (image_id, i, region)
            assert y2 <= H, (image_id, i, region)

            box = [x1, y1, x2, y2]
            boxes.append(box)

        boxes = np.array(boxes)

        return {
            'img_id': str(image_id),
            'img_fn': image_fn,
            'img': img,
            'boxes': boxes,
            # 'captions': cat_names
        }

def collate_fn(batch):
    img_ids = []
    imgs = []

    boxes = []

    # 另外一个脚本: refcocog_mattner.py 用了caption,是不是不用caption也可以？
    captions = []

    for i, entry in enumerate(batch):
        img_ids.append(entry['img_id'])
        imgs.append(entry['img'])
        boxes.append(entry['boxes'])
        # captions.append(entry['captions'])

    batch_out = {}
    batch_out['img_ids'] = img_ids
    batch_out['imgs'] = imgs

    batch_out['boxes'] = boxes

    # batch_out['captions'] = captions

    return batch_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--refcocoroot', type=str, default='/workspace/yfl/datasets/ref/')
    parser.add_argument('--cocoroot', type=str, default='/workspace/yfl/datasets/')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    refcoco_dir = Path(args.refcocoroot).resolve()
    refcocog_dir = refcoco_dir.joinpath('refcocog')
    coco_dir = Path(args.cocoroot).resolve()
    refcoco_images_dir = coco_dir.joinpath('train2014')
    dataset_name = 'RefCOCOg'

    out_dir = refcocog_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    dataset = RefCOCODataset(refcoco_dir, refcoco_images_dir, coco_dir, args.split)
    print('# Images:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_boxes_GT.h5')
    print('features will be saved at', output_fname)

    # DIM是2048
    desc = f'{dataset_name}_given_boxes_({DIM})'

    extract(output_fname, dataloader, desc)
