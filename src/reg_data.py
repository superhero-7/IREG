import json
import h5py
import torch
import random
import numpy as np

from pathlib import Path
from copy import deepcopy
from transformers import T5TokenizerFast
from modeling.tokenization import VLT5TokenizerFast
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from tools.vlt5_utils import xywh_to_xyxy, get_iou
from eval_utils.refcoco_utils import REFER
# from memory_profiler import profile


# 这个函数已经初步Clean过了，一个更清晰的办法就是把主函数中能抽象成小函数的部分都往外抽；
class RefCOCOGenerationFineTuneDataset(Dataset):
    def __init__(self, refer=None, split='train', rank=-1, topk=-1, verbose=True, args=None, mode='train', task=None):
        super().__init__()

        refcocog_feature_dir = Path(args.refcoco_dir)
        refcocog_feature_dir = refcocog_feature_dir.joinpath('features')

        # topk 一般是-1
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode
        self.task = task

        # Loading datasets to data
        self.split = split
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                # T5的一个extend
                self.tokenizer = VLT5TokenizerFast.from_pretrained(args.backbone)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(args.backbone)

        data = []
        if refer != None:
            self.refer = refer
        else:
            self.refer = REFER(args.dataset, args.dataset_split, verbose=True)
        ref_ids = self.refer.getRefIds(split=split)

        # 这个用法不就把所有的sentence都用上了，妙啊！
        for ref_id in ref_ids:

            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            refBox[2] = refBox[2] + refBox[0]
            refBox[3] = refBox[3] + refBox[1]
            if self.mode == 'train':
                for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                    caption = sent["raw"]
                    data.append(
                        {
                            "caption": caption,
                            "sent_id": sent_id,
                            "image_id": image_id,
                            "refBox": refBox,
                            "ref_id": ref_id,
                        }
                    )
            else:
                sent = random.choice(ref["sentences"])
                caption = sent["raw"]
                data.append(
                    {
                        "caption": caption,
                        "refBox": refBox,
                        "ref_id": ref_id,
                        "image_id": image_id,
                    }
                )

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        # topk 现在就是-1(没有指定就是)
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes

        # 有个问题train2014_obj36.h5一共有20几个G
        self.source_to_h5 = {
            'train': refcocog_feature_dir.joinpath('train_obj36.h5'),
            'train_target': refcocog_feature_dir.joinpath('train_target.h5'),
            'train_ann': refcocog_feature_dir.joinpath('train_ann.h5'),
            'val': refcocog_feature_dir.joinpath('val_obj36.h5'),
            'val_target': refcocog_feature_dir.joinpath('val_target.h5'),
            'val_ann': refcocog_feature_dir.joinpath('val_ann.h5'),
            'test': refcocog_feature_dir.joinpath('test_obj36.h5'),
            'test_target': refcocog_feature_dir.joinpath('test_target.h5'),
            'test_ann': refcocog_feature_dir.joinpath('test_ann.h5'),
            'testA': refcocog_feature_dir.joinpath('testA_obj36.h5'),
            'testA_target': refcocog_feature_dir.joinpath('testA_target.h5'),
            'testA_ann': refcocog_feature_dir.joinpath('testA_ann.h5'),
            'testB': refcocog_feature_dir.joinpath('testB_obj36.h5'),
            'testB_target': refcocog_feature_dir.joinpath('testB_target.h5'),
            'testB_ann': refcocog_feature_dir.joinpath('testB_ann.h5'),
        }

    def __len__(self):
        return len(self.data)

    # @profile(precision=4,stream=open('memory_profiler.log','w+')) 
    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = deepcopy(self.data[idx])
        ref_id = datum['ref_id']
        out_dict['ref_id'] = ref_id
        out_dict['image_id'] = datum['image_id']
        out_dict['refBox'] = datum['refBox']

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['image_id']
            out_dict['img_id'] = img_id

            source = self.split
            source_target = source + '_target'

            f = self.source_to_h5[source]
            f_target = self.source_to_h5[source_target]

            if isinstance(f, Path):
                f = h5py.File(f, 'r')
                self.source_to_h5[source] = f

            if isinstance(f_target, Path):
                f_target = h5py.File(f_target, 'r')
                self.source_to_h5[source_target] = f_target

            # 这个感觉其实也是可以不需要的
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]

            # 这边的n_boxes其实是用faster rcnn 提取出来的n_boxes(其实是36个)
            boxes = f[f'{img_id}/boxes'][:self.args.n_boxes]
            target_boxes = f_target[f'{ref_id}/boxes'][:]  # 其实就一个

            if self.args.shuffle_boxes and self.mode == 'train':
                box_indices = np.arange(len(boxes))
                np.random.shuffle(box_indices)

                boxes = boxes[box_indices]

            boxes_all = np.concatenate((boxes, target_boxes), axis=0)
            n_boxes = len(boxes_all)
            assert n_boxes == 37, ('Something is wrong! T_T', n_boxes)

            out_dict['n_boxes'] = n_boxes


            # Normalize the boxes (to 0 ~ 1)
            boxes_all[:, (0, 2)] /= img_w
            boxes_all[:, (1, 3)] /= img_h

            # 这个boxes不知道是干甚用，留一个疑问
            np.testing.assert_array_less(boxes_all, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes_all, 0+1e-5)
            boxes_all = torch.from_numpy(boxes_all)


            boxes_all.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes_all

            # 理论上有多少n_box就应该有多少个features
            feats = f[f'{img_id}/features'][:self.args.n_boxes]
            feats = torch.from_numpy(feats)

            if self.args.shuffle_boxes and self.mode == 'train':
                feats = feats[box_indices]

            feats_target = f_target[f'{ref_id}/features'][:]
            feats_target = torch.from_numpy(feats_target)
            feats_all = torch.cat((feats, feats_target), axis=0)

            # feats_all = torch.from_numpy(feats_all)

            out_dict['vis_feats'] = feats_all
            out_dict['boxes'] = boxes_all

            if self.args.use_mmi and self.mode == 'train':
                ref = self.refer.Refs[ref_id]
                target_class_id = ref['category_id']

                source_ann = source + '_ann'
                f_ann = self.source_to_h5[source_ann]

                if isinstance(f_ann, Path):
                    f_ann = h5py.File(f_ann, 'r')
                    self.source_to_h5[source_ann] = f_ann
                Anns = self.refer.imgToAnns[img_id]
                # 这里random.choice其实也不是很好，但是我之前没有保存class，所以这里没办法选类别
                # 而且可能选到自己...但有的数据就只有自己咋办呢？
                Anns_hard = []
                for ann in Anns:
                    if ann['category_id'] == target_class_id and ann['id'] != ref['ann_id']:
                        Anns_hard.append(ann)
                if Anns_hard:
                    neg_ann = random.choice(Anns_hard)
                else:
                    # 发现了个bug，之前直接random.choice一个还是有可能选到同一个框；
                    # 下面代码没有跑过不知道会不会出问题
                    while True:
                        neg_ann = random.choice(Anns)
                        if len(Anns) == 1:
                            break
                        else:
                            if neg_ann['id'] != ref['ann_id']:
                                break
                neg_ann_id = neg_ann['id']

                # 这里很有可能会有的neg_ref_id没有在f_target里边，报错了再回来改
                # 神奇地没有出bug！为什么没有出
                neg_target_boxes = f_ann[f'{neg_ann_id}/boxes']
                neg_boxes = np.concatenate((boxes, neg_target_boxes), axis=0)
                neg_boxes[:, (0, 2)] /= img_w
                neg_boxes[:, (1, 3)] /= img_h

                # assert_array_less: all element of the first object are strictly smaller than
                # those of the second object
                np.testing.assert_array_less(neg_boxes, 1 + 1e-5)
                # np.testing.assert_array_less(boxes, 1+5e-2)
                np.testing.assert_array_less(-neg_boxes, 0 + 1e-5)
                neg_boxes = torch.from_numpy(neg_boxes)

                neg_boxes.clamp_(min=0.0, max=1.0)

                neg_feats_target = f_ann[f'{neg_ann_id}/features']
                neg_feats_all = np.concatenate((feats, neg_feats_target), axis=0)
                neg_feats_all = torch.from_numpy(neg_feats_all)
                out_dict['boxes'] = torch.stack((out_dict['boxes'], neg_boxes), dim=0)
                out_dict['vis_feats'] = torch.stack((out_dict['vis_feats'], neg_feats_all), dim=0)

        ###### Text #####x

        visual_token = "<vis_extra_id_36>"

        #prefix = "refer expressions generation:"
        prefix = "caption region:"
        # prefix = "grounding:"
        input_text = f'{prefix} {visual_token}'

        # tokenizer都会把输入映射成input_ids
        # encode会在input_text后面加一个</s>
        input_ids = self.tokenizer.encode(input_text, max_length=self.args.max_text_length, truncation=True)
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['input_text'] = input_text
        if self.mode == 'train':
            sent = datum['caption']
            target_text = sent
            target_ids = self.tokenizer.encode(target_text, max_length=self.args.max_text_length, truncation=True)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
            out_dict['target_text'] = target_text
        else:
            out_dict['sent'] = datum['caption']
            # out_dict['bad_sent'] = datum['bad_sent']

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        args = self.args

        B = len(batch)

        if args.use_vision:
            V_L = max([b['n_boxes'] for b in batch])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            # 这种写法我已经见过了
            if args.use_mmi and self.mode == 'train':
                boxes = torch.zeros(B, 2, V_L, 4, dtype=torch.float)
                vis_feats = torch.zeros(B, 2, V_L, feat_dim, dtype=torch.float)
            else:
                boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
                vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

            # 特别是这个mask的写法，先初始化一个全零的矩阵，然后再往里面填！其实应该是不需要了
            # vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        # assert V_L == 38, '你小子又搞错啦！！！大笨蛋'
        assert feat_dim == 2048, '你小子又搞错啦！！！大笨蛋'

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        input_texts = []

        if self.mode == 'train':
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
            target_texts = []
        else:
            sents = []
        ref_ids = []
        image_ids = []
        refBoxes = []

        for i, entry in enumerate(batch):

            if args.use_vision:

                if args.use_mmi and self.mode == 'train':
                    boxes[i, 0, :entry['n_boxes']] += entry['boxes'][0]
                    vis_feats[i, 0, :entry['n_boxes']] += entry['vis_feats'][0]
                    boxes[i, 1, :entry['n_boxes']] += entry['boxes'][1]
                    vis_feats[i, 1, :entry['n_boxes']] += entry['vis_feats'][1]
                else:
                    boxes[i, :entry['n_boxes']] += entry['boxes']
                    vis_feats[i, :entry['n_boxes']] += entry['vis_feats']

                # vis_attention_mask[i, :entry['n_boxes']] = 1

            input_ids[i, :entry['input_length']] = entry['input_ids']
            input_texts.append(entry['input_text'])

            if self.mode == 'train':
                target_texts.append(entry['target_text'])
                target_ids[i, :entry['target_length']] = entry['target_ids']
            else:
                sents.append(entry['sent'])
            ref_ids.append(entry['ref_id'])
            image_ids.append(entry['image_id'])
            refBoxes.append(entry['refBox'])


        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['vis_attention_mask'] = vis_attention_mask

        batch_entry['input_ids'] = input_ids
        batch_entry['input_texts'] = input_texts

        if self.mode == 'train':
            # 下面首先把pad的地方变成-100，然后把不存在target的部分也变成-100
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100
            batch_entry['target_ids'] = target_ids
            batch_entry['target_texts'] = target_texts
        else:
            batch_entry['sents'] = sents

        batch_entry['ref_ids'] = ref_ids
        batch_entry['image_ids'] = image_ids
        batch_entry['refBoxes'] = refBoxes
        if self.task is None:
            batch_entry['task'] = 'reg'
        else:
            batch_entry['task'] = self.task

        # batch_entry['args'] = args

        return batch_entry

# 完成基础Clean, 同上的优化方式就是把主函数中的一些功能抽象成小函数
class DialogREGDataset(Dataset):
    def __init__(self, refer=None, split='train', rank=-1, topk=-1, verbose=True, args=None, mode='train', task=None):
        super().__init__()

        refcocog_feature_dir = Path(args.refcoco_dir)
        refcocog_feature_dir = refcocog_feature_dir.joinpath('features')

        # topk 一般是-1
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode
        self.task = task

        # Loading datasets to data
        self.split = split
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                # T5的一个extend
                self.tokenizer = VLT5TokenizerFast.from_pretrained(args.backbone)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(args.backbone)

        # read bad sents as supply information, plug into bellow data list
        print("Load the bad referring expressions.")
        with open(self.args.bad_res_path, 'r') as fp:
            bad_sents = json.load(fp)


        # 实际上这个coco_img_dir是错的，根本不存在，但是好像不影响
        data = []
        if refer != None:
            self.refer = refer
        else:
            self.refer = REFER(args.dataset, args.dataset_split, verbose=True)
        # ref_ids = self.refer.getRefIds(split=split)

        for item in bad_sents:
            ref_id = item['ref_id']
            uni_id = item['uni_id']
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            refBox[2] = refBox[2] + refBox[0]
            refBox[3] = refBox[3] + refBox[1]
            if self.mode == 'train':
                for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                    caption = sent["raw"]
                    data.append(
                        {
                            "caption": caption,
                            "bad_sent": item['sent'],
                            "sent_id": sent_id,
                            "image_id": image_id,
                            "refBox": refBox,
                            "ref_id": ref_id,
                            'uni_id': uni_id,
                        }
                    )

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        # topk 现在就是-1(没有指定就是)
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes

        # 有个问题train2014_obj36.h5一共有20几个G
        self.source_to_h5 = {
            'train': refcocog_feature_dir.joinpath('train_obj36.h5'),
            'train_target': refcocog_feature_dir.joinpath('train_target.h5'),
            'train_ann': refcocog_feature_dir.joinpath('train_ann.h5'),
            'val': refcocog_feature_dir.joinpath('val_obj36.h5'),
            'val_target': refcocog_feature_dir.joinpath('val_target.h5'),
            'val_ann': refcocog_feature_dir.joinpath('val_ann.h5'),
            'test': refcocog_feature_dir.joinpath('test_obj36.h5'),
            'test_target': refcocog_feature_dir.joinpath('test_target.h5'),
            'test_ann': refcocog_feature_dir.joinpath('test_ann.h5'),
            'testA': refcocog_feature_dir.joinpath('testA_obj36.h5'),
            'testA_target': refcocog_feature_dir.joinpath('testA_target.h5'),
            'testA_ann': refcocog_feature_dir.joinpath('testA_ann.h5'),
            'testB': refcocog_feature_dir.joinpath('testB_obj36.h5'),
            'testB_target': refcocog_feature_dir.joinpath('testB_target.h5'),
            'testB_ann': refcocog_feature_dir.joinpath('testB_ann.h5'),
            'train_ofa_target': refcocog_feature_dir.joinpath('train_bad_ofa_target_MM.h5'),
            'testA_ofa_target': refcocog_feature_dir.joinpath('testA_ofa_target.h5'),
            'testB_ofa_target': refcocog_feature_dir.joinpath('testB_ofa_target.h5'),
            'test_ofa_target': refcocog_feature_dir.joinpath('test_ofa_target.h5'),
            'val_ofa_target': refcocog_feature_dir.joinpath('val_ofa_target.h5'),
        }

    def __len__(self):
        return len(self.data)

    # @profile(precision=4,stream=open('memory_profiler.log','w+')) 
    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = deepcopy(self.data[idx])
        uni_id = datum['uni_id']
        ref_id = datum['ref_id']
        out_dict['ref_id'] = ref_id
        out_dict['image_id'] = datum['image_id']
        out_dict['refBox'] = datum['refBox']
        # out_dict['refBox'][2] = out_dict['refBox'][2] + out_dict['refBox'][0]
        # out_dict['refBox'][3] = out_dict['refBox'][3] + out_dict['refBox'][1]
        # uid = datum['uid']
        # out_dict['uid'] = uid

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['image_id']
            out_dict['img_id'] = img_id

            source = self.split
            source_target = source + '_target'

            f = self.source_to_h5[source]
            f_target = self.source_to_h5[source_target]

            if isinstance(f, Path):
                f = h5py.File(f, 'r')
                self.source_to_h5[source] = f

            if isinstance(f_target, Path):
                f_target = h5py.File(f_target, 'r')
                self.source_to_h5[source_target] = f_target

            # 视觉特征加入 For Dialog Training
            source_ofa_target = source + '_ofa_target'
            f_ofa_target = self.source_to_h5[source_ofa_target]
            if isinstance(f_ofa_target, Path):
                f_ofa_target = h5py.File(f_ofa_target, 'r')
                self.source_to_h5[source_ofa_target] = f_ofa_target


            # 这个感觉其实也是可以不需要的
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]

            # 这边的n_boxes其实是用faster rcnn 提取出来的n_boxes(其实是36个)
            boxes = f[f'{img_id}/boxes'][:self.args.n_boxes]
            target_boxes = f_target[f'{ref_id}/boxes'][:]  # 其实就一个
            # 视觉特征加入 For Dialog Training
            ofa_target_boxes = f_ofa_target[f'{uni_id}/boxes'][:]
            # shuffle box order
            if self.args.shuffle_boxes and self.mode == 'train':
                box_indices = np.arange(len(boxes))
                np.random.shuffle(box_indices)

                boxes = boxes[box_indices]

            if self.mode=='train':
                boxes_all = np.concatenate((boxes, target_boxes, ofa_target_boxes), axis=0)
                n_boxes = len(boxes_all)
                assert n_boxes == 38, ('Something is wrong! T_T', n_boxes)
            else:
                boxes_all = np.concatenate((boxes, target_boxes), axis=0)
                n_boxes = len(boxes_all)
                assert n_boxes == 37, ('Something is wrong! T_T', n_boxes)

            out_dict['n_boxes'] = n_boxes


            # Normalize the boxes (to 0 ~ 1)
            boxes_all[:, (0, 2)] /= img_w
            boxes_all[:, (1, 3)] /= img_h

            # 这个boxes不知道是干甚用，留一个疑问
            np.testing.assert_array_less(boxes_all, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes_all, 0+1e-5)
            boxes_all = torch.from_numpy(boxes_all)

            boxes_all.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes_all

            # 理论上有多少n_box就应该有多少个features
            feats = f[f'{img_id}/features'][:self.args.n_boxes]
            feats = torch.from_numpy(feats)

            if self.args.shuffle_boxes and self.mode == 'train':
                feats = feats[box_indices]

            if self.mode=='train':
                feats_target = f_target[f'{ref_id}/features'][:]
                feats_target = torch.from_numpy(feats_target)
                feats_ofa_target = f_ofa_target[f'{uni_id}/features'][:]
                feats_ofa_target = torch.from_numpy(feats_ofa_target)
                feats_all = torch.cat((feats, feats_target, feats_ofa_target), axis=0)
            else:
                feats_target = f_target[f'{ref_id}/features'][:]
                feats_target = torch.from_numpy(feats_target)
                feats_all = torch.cat((feats, feats_target), axis=0)

            # feats_all = torch.from_numpy(feats_all)

            out_dict['vis_feats'] = feats_all
            out_dict['boxes'] = boxes_all

        ###### Text #####x

        visual_token = "<vis_extra_id_36>"

        #prefix = "refer expressions generation:"
        prefix = "caption region:"
        # prefix = "grounding:"

        if self.mode == 'train':
            unlocated = "incorrectly unlocated as"
            wrong_visual_token = "<vis_exra_id_37>"
            refine = "Please refine it:"
            bad_sent = datum['bad_sent']
            input_text = f'{prefix} {visual_token} {bad_sent} {unlocated} {wrong_visual_token} {refine}'
        else:
            input_text = f'{prefix} {visual_token}'

        # tokenizer都会把输入映射成input_ids
        # encode会在input_text后面加一个</s>
        input_ids = self.tokenizer.encode(input_text, max_length=self.args.max_text_length, truncation=True)
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['input_text'] = input_text
        if self.mode == 'train':
            sent = datum['caption']
            target_text = sent
            target_ids = self.tokenizer.encode(target_text, max_length=self.args.max_text_length, truncation=True)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
            out_dict['target_text'] = target_text
        else:
            out_dict['sent'] = datum['caption']
            # out_dict['bad_sent'] = datum['bad_sent']

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        args = self.args

        B = len(batch)

        if args.use_vision:
            V_L = max([b['n_boxes'] for b in batch])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            # 这种写法我已经见过了
            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

            # 特别是这个mask的写法，先初始化一个全零的矩阵，然后再往里面填！其实应该是不需要了
            # vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        # assert V_L == 38, '你小子又搞错啦！！！大笨蛋'
        assert feat_dim == 2048, '你小子又搞错啦！！！大笨蛋'

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        input_texts = []

        if self.mode == 'train':
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
            target_texts = []
        else:
            sents = []
        ref_ids = []
        image_ids = []
        refBoxes = []

        for i, entry in enumerate(batch):

            if args.use_vision:

                boxes[i, :entry['n_boxes']] += entry['boxes']
                vis_feats[i, :entry['n_boxes']] += entry['vis_feats']

                # vis_attention_mask[i, :entry['n_boxes']] = 1

            input_ids[i, :entry['input_length']] = entry['input_ids']
            input_texts.append(entry['input_text'])

            if self.mode == 'train':
                target_texts.append(entry['target_text'])
                target_ids[i, :entry['target_length']] = entry['target_ids']
            else:
                sents.append(entry['sent'])
            ref_ids.append(entry['ref_id'])
            image_ids.append(entry['image_id'])
            refBoxes.append(entry['refBox'])


        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['vis_attention_mask'] = vis_attention_mask

        batch_entry['input_ids'] = input_ids
        batch_entry['input_texts'] = input_texts

        if self.mode == 'train':
            # 下面首先把pad的地方变成-100，然后把不存在target的部分也变成-100
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100
            batch_entry['target_ids'] = target_ids
            batch_entry['target_texts'] = target_texts
        else:
            batch_entry['sents'] = sents

        batch_entry['ref_ids'] = ref_ids
        batch_entry['image_ids'] = image_ids
        batch_entry['refBoxes'] = refBoxes
        if self.task is None:
            batch_entry['task'] = 'reg'
        else:
            batch_entry['task'] = self.task

        # batch_entry['args'] = args

        return batch_entry

def get_loader(args, refer=None, split='train', mode='train', task=None,
               batch_size=32, workers=4, distributed=False, gpu=0,
               dialog_sp_train=False,topk=-1):

    verbose = (gpu == 0)

    if dialog_sp_train:
        dataset = DialogREGDataset(
            refer=refer,
            split=split,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            task=task,
        )
    else:
        dataset = RefCOCOGenerationFineTuneDataset(
            refer=refer,
            split=split,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode,
            task=task,
        )

    if distributed and mode == 'train':
        sampler = DistributedSampler(dataset)
    elif distributed and not (mode == 'train'):
        print("Get val sampler!")
        sampler = DistributedSampler(dataset, drop_last=True)
    else:
        sampler = None

    if mode == 'train':
        # shuffle与sampler是不能共存的
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=True)

    if task is None:
        loader.task = 'reg'
    else:
        loader.task = task
    loader.split_name = split
    #loader.evaluator = REGEvaluator()

    return loader