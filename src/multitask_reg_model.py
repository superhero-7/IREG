import torch
import numpy as np

import cv2
import sys
import os.path as osp

feature_extraction_dir = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'misc', 'feature_extraction')
if feature_extraction_dir not in sys.path:
    sys.path.insert(0, feature_extraction_dir)

from detectron2_given_target_box_maxnms import doit, build_model

from modeling.modeling_t5 import VLT5

class VLT5REG(VLT5):
    def __init__(self, config):
        super().__init__(config)

    # 这个函数是正常运行的 √ ^_^!!!
    # @profile
    def train_step(self, batch):

        device = next(self.parameters()).device

        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result
    
    # 这个函数是正常工作的 ^_^ !
    def test_step(self, batch, dialog_training, rewarder, dialog_round=2, last_round=False, threshold=0.5, basemodel=None, detector=None, img_dir=None, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        bs = len(ref_ids)

        prefix = "caption region:"
        visual_token = "<vis_extra_id_36>"
        unlocated = "incorrectly unlocated as"
        wrong_visual_token = "<vis_exra_id_37>"            
        refine = "Please refine it:"

        if dialog_training:

            dialog_generatae_sents = [[''] * dialog_round for _ in range(bs)]  # size: bs*num_dialog_round
            dialog_generatae_sents_ofa_ious = [[-1] * dialog_round for _ in range(bs)]
            dialog_generatae_ofa_box = [[] for _ in range(bs)]
            for dialog_round_idx in range(dialog_round):

                if dialog_round_idx==0:
                    output = basemodel.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(new_vis_feats, new_vis_pos),
                        **kwargs
                    )

                output_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)  # bs*sentence_len
                for bs_idx, output_sent in enumerate(output_sents):
                    dialog_generatae_sents[bs_idx][dialog_round_idx] = output_sent
                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']
                sample_dict['sents'] = output_sents  # a list of sent
                # rewarder should return a tensor in the shape of bacthsize
                # sample_rewards: (batch_size, 1)
                sample_rewards, sample_rewards_mask, det_result = rewarder.compute_score(sample_dict)
                ofa_box = [det_result[0]['box']]
                dialog_generatae_ofa_box[0].append(ofa_box)
                
                for bs_idx, sample_reward in enumerate(sample_rewards):
                    dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = sample_reward.item()
                # IOU surpass 0.5, the we think it located the target object.
                if sample_rewards[0] >= threshold:
                    break
                # update input ids
                # 这个地方其实也是需要bs_idx，但因为测试的bs只会是1所以不影响；
                img_path = img_dir+'/COCO_train2014_' + str(batch['image_ids'][0]).zfill(12) + '.jpg'
                img = cv2.imread(img_path)
                instances, ofa_feature = doit(img, np.array(ofa_box), detector)
                # ofa_feature = torch.from_numpy(ofa_feature)
                new_vis_feats = torch.cat((vis_feats[0], ofa_feature), axis=0)
                new_vis_feats = new_vis_feats.unsqueeze(0)
                ofa_box = torch.tensor(ofa_box).to(device)
                new_vis_pos = torch.cat((vis_pos[0], ofa_box), axis=0)
                new_vis_pos = new_vis_pos.unsqueeze(0)
                input_text = f'{prefix} {visual_token} {output_sents[0]} {unlocated} {wrong_visual_token} {refine}'
                input_ids = self.tokenizer.encode(input_text)
                input_ids = torch.LongTensor(input_ids).to(device)
                input_ids = input_ids.unsqueeze(0)

            if last_round:
                generated_sents = output_sents

            result = []
            for bs_idx, sent in enumerate(generated_sents):
                result.append(
                    {
                        'ref_id': ref_ids[bs_idx],
                        'sent': sent,
                        'dialog_generate_sent': dialog_generatae_sents[bs_idx],
                        'dialog_generate_sent_ofa_iou': dialog_generatae_sents_ofa_ious[bs_idx],
                        'dialog_generatae_ofa_box': dialog_generatae_ofa_box[bs_idx]
                    }
                )

            return result
        else:
            # generate 可以指定num_beams, 以及num_return_sequence(default=1), so here return only 1 sentence for 1 ref_id！
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )

            # this is a list type, length equal to batch size,
            # e.g.['A giraffe standing in the shade of a tree.','A giraffe standing in the middle of two other giraffes.', ...]
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

            result = []
            for i, sent in enumerate(generated_sents):

                result.append(
                    {
                        'ref_id': ref_ids[i],
                        'sent': sent,
                    }
                )

            return result