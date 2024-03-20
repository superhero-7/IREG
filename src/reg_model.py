import torch
from torch.nn import CrossEntropyLoss

from copy import deepcopy
from types import MethodType
from undecorated import undecorated
# from memory_profiler import profile

from eval_utils.cider.cider import Cider
from eval_utils.tokenizer.ptbtokenizer import PTBTokenizer

from modeling.modeling_t5 import VLT5

# import sys
# import os.path as osp
# if osp.join('/raid_sda/yfl/codebase/VL-T5-REG/feature_extraction') not in sys.path:
#     sys.path.insert(0, osp.join('/raid_sda/yfl/codebase/VL-T5-REG/feature_extraction'))

# from detectron2_given_target_box_maxnms import doit, build_model
# cv2.setNumThreads(0)

class VLT5REG(VLT5):
    def __init__(self, config):
        super().__init__(config)

    # MMI和普通训练的都是可以正常工作的 √
    def train_step(self, batch, use_mmi=False, epoch=None, lama=1, margin=0.5, use_negative_text_training=False):

        device = next(self.parameters()).device
        if use_mmi:
            vis_feats = torch.squeeze(batch['vis_feats'][:, 0].to(device))
            vis_pos = torch.squeeze(batch['boxes'][:, 0].to(device))

            neg_vis_feats = torch.squeeze(batch['vis_feats'][:, 1].to(device))
            neg_vis_pos = torch.squeeze(batch['boxes'][:, 1].to(device))

            input_ids = batch['input_ids'][:].to(device)

            lm_labels = batch["target_ids"].to(device)

            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            neg_output = self(
                input_ids=input_ids,
                vis_inputs=(neg_vis_feats, neg_vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            lm_mask = lm_labels != -100
            B, L = lm_labels.size()

            pos_loss = output['loss']
            neg_loss = neg_output['loss']

            # 这里一会改还不知道能不能跑起来...
            if epoch % 80 == 0:
                margin /= 2
            loss = pos_loss + lama * (max(0, margin + pos_loss - neg_loss))

            result = {
                'loss': loss
            }
            return result
        elif use_negative_text_training:
            vis_feats = batch['vis_feats'].to(device)
            input_ids = batch['input_ids'].to(device)
            vis_pos = batch['boxes'].to(device)

            lm_labels = batch["target_ids"].to(device)
            negative_labels = batch["negative_sent_ids"].to(device)

            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            # 这个地方可以优化计算但是我现在先不优化
            neg_output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=negative_labels,
                reduce_loss=reduce_loss
            )

            lm_mask = lm_labels != -100
            B, L = lm_labels.size()

            pos_loss = output['loss']
            neg_loss = neg_output['loss']

            # 这里一会改还不知道能不能跑起来...
            # if epoch % 10 == 0:
            #     margin /= 2
            loss = pos_loss + lama * (max(0, margin + pos_loss - neg_loss))
            # import pdb
            # pdb.set_trace()
            # loss = torch.mean(pos_loss + lama * (torch.clamp(margin + pos_loss - neg_loss, min=0.0)))

            result = {
                'loss': loss
            }
            return result
        else:
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
    
    # 这个函数也是正常工作的 √    
    def rec_rl_train_step(self, batch, rewarder, use_combine=False, lamda=0.5, combine_with_celoss=False):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none', ignore_index=0)
        rewarder = rewarder
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size
        bs = len(target_sents)

        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        sample_output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            max_length=20,
        )
        sample_sents = self.tokenizer.batch_decode(sample_output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        # original scores: tuple: (tensor_matrix1, ..., tensor_matrixT) tensor_matrix_i: batch_size*vocab_size
        scores = torch.stack(sample_output.scores, dim=0).permute(1, 0, 2)  # batch_size*sentence_len*vocabulary
        scores = scores.reshape(-1, scores.size(-1))
        target = sample_output.sequences[:, 1:].reshape(-1)
        # index = target != 0
        # print(scores[list(range(len(scores))), target[index]])

        loss = criterion(scores,
                         target,
                         )
        loss = loss.view(bs, -1)
        loss = torch.mean(loss, dim=1)

        sample_dict = {}
        sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
        sample_dict['refBoxes'] = batch['refBoxes']
        sample_dict['sents'] = sample_sents  # a list of sent
        # rewarder should return a tensor in the shape of bacthsize
        sample_rewards, sample_rewards_mask, _ = rewarder.compute_score(sample_dict)
        # sample_rewards = torch.from_numpy(sample_rewards).to(device)

        greedy_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            do_sample=False,
            max_length=20,
        )
        greedy_sents = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        greedy_dict = {}
        greedy_dict['image_ids'] = batch['image_ids']
        greedy_dict['refBoxes'] = batch['refBoxes']
        greedy_dict['sents'] = greedy_sents
        reward_baseline, reward_baseline_mask, _ = rewarder.compute_score(greedy_dict)

        # try to put greedy_dict and sample_dict together
        # sadly,this will out of memory
        # dict = {}
        # dict['image_ids'] = batch['image_ids'] + batch['image_ids']
        # dict['refBoxes'] = batch['refBoxes'] + batch['refBoxes']
        # dict['sents'] = sample_sents + greedy_sents
        # rewards, masks = rewarder.compute_score(dict)
        # sample_rewards = rewards[:bs]
        # reward_baseline = rewards[bs:]
        # sample_rewards_mask = masks[:bs]
        # reward_baseline_mask = masks[bs:]

        # The code below maybe need may be not, I think keep it will be better
        # reward_baseline = torch.from_numpy(greedy_rewards).to(device)
        # print(output_rewards.size(), reward_baseline.size())
        # 即两个mask都是false的时候，即 ious 值都没有超过0.5时，将其mask掉
        # final_reward_mask = sample_rewards_mask | reward_baseline_mask
        # reward = torch.clamp((sample_rewards-reward_baseline)*final_reward_mask, min=0.0)
        # reward = torch.exp(torch.clamp(torch.tensor(1.5)*(sample_rewards-reward_baseline)*final_reward_mask, min=0.0))-torch.tensor(1.0)
        reward = torch.clamp((sample_rewards-reward_baseline), min=0.0)
        # reward = torch.exp(torch.clamp(torch.tensor(2.0)*(sample_rewards - reward_baseline), min=0.0)) - torch.tensor(1.0)
        # reward = sample_rewards-reward_baseline

        # reward = torch.clamp((sample_rewards-torch.tensor(0.5)), min=0.0)
        # reward = sample_rewards*sample_rewards_mask
        # reward = sample_rewards-torch.tensor(0.5)

        if use_combine:
            cider = Cider()
            sample_sents_dict = {}
            for ref_id, output_sent in zip(list(range(len(sample_sents))), sample_sents):
                sample_sents_dict[str(ref_id)] = [output_sent]
            target_sents_dict = {}
            for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
                target_sents_dict[str(ref_id)] = [target_sent]
            tokenizer = PTBTokenizer()
            sample_sents_dict = tokenizer.tokenize(sample_sents_dict)
            target_sents_dict = tokenizer.tokenize(target_sents_dict)

            sample_cider_reward, sample_cider_rewards = cider.compute_score(target_sents_dict, sample_sents_dict)
            sample_cider_rewards = torch.from_numpy(sample_cider_rewards).to(device)

            greedy_sents_dict = {}
            for idx, greedy_sent in zip(list(range(len(greedy_sents))), greedy_sents):
                greedy_sents_dict[str(idx)] = [greedy_sent]

            greedy_cider_reward, greedy_cider_rewards = cider.compute_score(target_sents_dict, greedy_sents_dict)
            reward_cider_baseline = torch.from_numpy(greedy_cider_rewards).to(device)
            reward = lamda*reward + (1-lamda)*torch.clamp((sample_cider_rewards-reward_cider_baseline), min=0.0)
            reslut['sample_cider_reward'] = sample_cider_rewards.mean()

        loss = reward*loss
        loss = loss.mean()

        if combine_with_celoss:
            lm_labels = batch["target_ids"].to(device)
            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )
            cross_entropy_loss = output['loss']

            loss += 0.1*cross_entropy_loss
        reslut['loss'] = loss
        # reslut['reward_baseline'] = reward_baseline.mean()
        reslut['sample_reward'] = sample_rewards.mean()
        reslut['reward'] = reward.mean()

        return reslut

    # 这个函数是正常工作的 √
    def rl_train_step(self, batch):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none', ignore_index=0)
        rewarder = Cider()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size
        bs = len(target_sents)


        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            max_length=20,
        )
        output_sents = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        # original scores: tuple: (tensor_matrix1, ..., tensor_matrixT) tensor_matrix_i: batch_size*vocab_size
        scores = torch.stack(output.scores, dim=0).permute(1, 0, 2)  # batch_size*sentence_len*vocabulary
        # probs = torch.nn.functional.softmax(scores, dim=-1)
        scores = scores.reshape(-1, scores.size(-1))  # (batch_size*sentence_len, vocabulary)
        target = output.sequences[:, 1:].reshape(-1)  # (batch_size*sentence_len)
        # index = target != 0
        # print(scores[list(range(len(scores))), target[index]])

        # here loss is a vector which length is batch_size*sentence_len
        loss = criterion(scores,
                         target,
                         )
        loss = loss.view(bs, -1)  # (batch_size, sentence_len)
        loss = torch.mean(loss, dim=1)

        output_sents_dict = {}
        for ref_id, output_sent in zip(list(range(len(output_sents))), output_sents):
            output_sents_dict[str(ref_id)] = [output_sent]
        # print('output_sents_dict:', len(output_sents_dict))
        target_sents_dict = {}
        for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
            target_sents_dict[str(ref_id)] = [target_sent]
        # print('target_sent_dict', len(target_sents_dict))
        # It seems change nothing bt PTBTokenizer,emmmmmm还是有点用的
        tokenizer = PTBTokenizer()
        output_sents_dict = tokenizer.tokenize(output_sents_dict)
        target_sents_dict = tokenizer.tokenize(target_sents_dict)
        # print('output_sents_dict after tokenize', len(output_sents_dict))
        # print('target_sents_dict after tokenize', len(target_sents_dict))

        output_reward, output_rewards = rewarder.compute_score(target_sents_dict, output_sents_dict)
        output_rewards = torch.from_numpy(output_rewards).to(device)
        # print('output_rewards:', len(output_rewards))
        # logits_warper = LogitsProcessorList([])

        greedy_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            do_sample=False,
            max_length=20,
        )
        greedy_sents = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        greedy_sents_dict = {}
        for idx, greedy_sent in zip(list(range(len(greedy_sents))), greedy_sents):
            greedy_sents_dict[str(idx)] = [greedy_sent]

        greedy_reward, greedy_rewards = rewarder.compute_score(target_sents_dict, greedy_sents_dict)
        reward_baseline = torch.from_numpy(greedy_rewards).to(device)
        # print(output_rewards.size(), reward_baseline.size())
        # I think here maybe need a little change, get the every reward bigger than zero
        # reward = torch.clamp((output_rewards-reward_baseline), min=0.0)
        reward = output_rewards - reward_baseline
        loss = reward*loss
        loss = loss.mean()
        reslut['loss'] = loss
        reslut['sample_reward'] = output_rewards.mean()
        reslut['reward_baseline'] = reward_baseline.mean()
        reslut['reward'] = reward.mean()

        return reslut

    # 没测试过
    def rl_train_step2(self, batch):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none')
        rewarder = Cider()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size


        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True
        )
        output_sents = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        scores = torch.stack(output.scores, dim=0).permute(1, 0, 2)
        loss = criterion(scores.reshape(-1, scores.size(-1)),
                         output.sequences[:, 1:].reshape(-1),
                         )
        loss = loss.view(len(target_sents), -1)
        loss = torch.mean(loss, dim=1)

        output_sents_dict = {}
        for ref_id, output_sent in zip(list(range(len(output_sents))), output_sents):
            output_sents_dict[str(ref_id)] = [output_sent]
        # print('output_sents_dict:', len(output_sents_dict))
        target_sents_dict = {}
        for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
            target_sents_dict[str(ref_id)] = [target_sent]
        # print('target_sent_dict', len(target_sents_dict))
        # It seems change nothing bt PTBTokenizer,emmmmmm还是有点用的
        tokenizer = PTBTokenizer()
        output_sents_dict = tokenizer.tokenize(output_sents_dict)
        target_sents_dict = tokenizer.tokenize(target_sents_dict)
        # print('output_sents_dict after tokenize', len(output_sents_dict))
        # print('target_sents_dict after tokenize', len(target_sents_dict))

        output_reward, output_rewards = rewarder.compute_score(target_sents_dict, output_sents_dict)
        output_rewards = torch.from_numpy(output_rewards).to(device)
        # print('output_rewards:', len(output_rewards))

        beam_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            num_beams=5,
            num_return_sequences=5,
            max_length=20,
        )
        beam_sents = self.tokenizer.batch_decode(beam_output, skip_special_tokens=True)
        beam_target_sents = []
        for target_sent in target_sents:
            beam_target_sents += [target_sent]*5

        beam_sents_dict = {}
        for idx, beam_sent in zip(list(range(len(beam_sents))), beam_sents):
            beam_sents_dict[str(idx)] = [beam_sent]

        beam_target_sents_dict = {}
        for idx, beam_target_sent in zip(list(range(len(beam_target_sents))), beam_target_sents):
            beam_target_sents_dict[str(idx)] = [beam_target_sent]
        beam_reward, beam_rewards = rewarder.compute_score(beam_target_sents_dict, beam_sents_dict)
        beam_rewards = torch.from_numpy(beam_rewards).to(device)
        beam_rewards = beam_rewards.view(-1, 5)
        reward_baseline = torch.mean(beam_rewards, dim=1)
        # print(output_rewards.size(), reward_baseline.size())
        # I think here maybe need a little change, get the every reward bigger than zero
        loss = torch.maximum((output_rewards-reward_baseline), torch.tensor(0))*loss
        loss = loss.mean()
        reslut['loss'] = loss

        return reslut

    # 这个函数是正常工作的 ^_^ √
    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def test_step(self, batch,**kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = deepcopy(batch['ref_ids'])

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
                    'sent': deepcopy(sent),
                }
            )

        return result

    # 这个函数是正常工作的 ^_^ √
    def test_step_for_bad_re_collection(self, batch, rewarder, threshold=0.5, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        bs = len(ref_ids)

        # generate 可以指定num_beams, 以及num_return_sequence(default=1), so here return only 1 sentence for 1 ref_id！
        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        # this is a list type, length equal to batch size,
        # e.g.['A giraffe standing in the shade of a tree.','A giraffe standing in the middle of two other giraffes.', ...]
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        sample_dict = {}
        sample_dict['image_ids'] = [item for item in batch['image_ids'] for i in range(5)]
        sample_dict['refBoxes'] = [item for item in batch['refBoxes'] for i in range(5)]
        sample_dict['sents'] = generated_sents

        sample_rewards, sample_rewards_mask, ofa_results = rewarder.compute_score(sample_dict, threshold=threshold)

        ofa_bboxes = []
        for ofa_result in ofa_results:
            ofa_bboxes.append(ofa_result['box'])

        result = {
            'sents': generated_sents,
            'masks': sample_rewards_mask,
            'boxes': ofa_bboxes,
        }

        return result
