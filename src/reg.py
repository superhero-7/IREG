# for OFA
from fairseq import utils,tasks
from fairseq import checkpoint_utils

import os
import sys
import os.path as osp

OFA_dir = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'OFA')
if OFA_dir not in sys.path:
    sys.path.insert(0, OFA_dir)

from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from data.mm_data.refcoco_dataset import collate
# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)

import json
import wandb
import shutil
import logging
import collections
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from pprint import pprint
from pprint import pformat
from reg_model import VLT5REG
from packaging import version
from reg_data import get_loader
from eval_utils.refcoco_utils import REFER
from torch.nn.parallel import DistributedDataParallel as DDP

from tools import dist_utils
from tools.param import parse_args
from tools.trainer_base import TrainerBase
from tools.vlt5_utils import load_state_dict, LossMeter, set_global_logging_level, count_parameters
# from memory_profiler import profile

from eval_utils.refEvaluation import RefEvaluation

set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent

# from detectron2_given_target_box_maxnms import doit, build_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class Critic():

    def __init__(self, args):

        self.args = args
        
        # turn on cuda if GPU is available
        self.use_cuda = torch.cuda.is_available()
        # use fp16 only when GPU is available
        self.use_fp16 = False

        # Load pretrained ckpt & config
        overrides = {"bpe_dir": str(OFA_dir) + "/utils/BPE"}
        # overrides = {"bpe_dir": "/home/yfl/OFA/utils/BPE"}
        if self.args.dataset in ['refcoco', 'refcocog']:
            ofa_checkepoint_path = args.ofa_ckpt_dir + \
                               self.args.dataset+'_base_best.pt'
        elif self.args.dataset =='refcoco+':
            ofa_checkepoint_path = args.ofa_ckpt_dir + \
                                   'refcocoplus' + '_base_best.pt'

        print("Load OFA model from:{}".format(ofa_checkepoint_path))
        self.models, self.cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(ofa_checkepoint_path),
            arg_overrides=overrides
        )

        self.cfg.common.seed = 7
        self.cfg.generation.beam = 5
        self.cfg.generation.min_len = 4
        self.cfg.generation.max_len_a = 0
        self.cfg.generation.max_len_b = 4
        self.cfg.generation.no_repeat_ngram_size = 3
        self.patch_image_size = self.cfg.task.patch_image_size

        # Fix seed for stochastic decoding
        if self.cfg.common.seed is not None and not self.cfg.generation.no_seed_provided:
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Move models to GPU 这里到到后面分布式可能需要指定move到哪个GPU上
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # Image transform
        from torchvision import transforms

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()
        self.eos_idx = self.task.src_dict.eos()  # 2

    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text.lower()),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    # Construct input for refcoco task

    def construct_sample(self, image: Image, text: str, img_id: int, region_coords):
        w, h = image.size
        w_resize_ratio = torch.tensor(self.patch_image_size / w)
        h_resize_ratio = torch.tensor(self.patch_image_size / h)
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])
        source = self.encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True,
                                  append_eos=True).unsqueeze(0)
        sample = {
            "id": str(img_id),
            "source": source[0],
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "w_resize_ratio": w_resize_ratio,
            "h_resize_ratio": h_resize_ratio,
            "region_coord": torch.tensor(region_coords),
        }
        return sample

    # Function to turn FP32 to FP16
    def apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def compute_score(self, dict, ofa_test=False, threshold=0.5):
        image_ids = dict['image_ids']
        refBoxes = dict['refBoxes']
        sents = dict['sents']

        assert len(image_ids) == len(refBoxes) == len(sents), 'Length does not match!'

        samples = []

        for image_id, refBox, sent in zip(image_ids, refBoxes, sents):
            img_path = self.args.img_dir + '/COCO_train2014_' + str(image_id).zfill(12) + '.jpg'
            # img_path = '/home/yfl/datasets/train2014/COCO_train2014_' + str(image_id).zfill(12) + '.jpg'
            image = Image.open(img_path)
            sample = self.construct_sample(image, sent, image_id, refBox)
            # 我怀疑这个地方会有内存泄漏的问题
            samples.append(deepcopy(sample))
        ofa_batch = collate(samples, self.pad_idx, self.eos_idx)
        ofa_batch = utils.move_to_cuda(ofa_batch) if self.use_cuda else ofa_batch
        ofa_batch = utils.apply_to_sample(self.apply_half, ofa_batch) if self.use_fp16 else ofa_batch
        with torch.no_grad():
            if ofa_test:
                result, scores = eval_step(self.task, self.generator, self.models, ofa_batch, ofa_test=ofa_test, threshold=threshold)
            else:
                result, scores, scores_mask = eval_step(self.task, self.generator, self.models, ofa_batch, ofa_test=ofa_test, threshold=threshold)

        if ofa_test:
            return scores, [], result
        else:
            return scores, scores_mask, result


class Trainer(TrainerBase):
    # @profile
    def __init__(self, args, train_loader=None, val_loader=None, test_loaderA=None, test_loaderB=None, train=True, refer=None):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loaderA=test_loaderA,
            test_loaderB=test_loaderB,
            train=train)

        self.wandb_initialized = False

        assert refer != None, "Please assign refer to Trainer!"
        self.refer = refer

        # 初始化模型
        model_kwargs = {}
        self.model_class = VLT5REG
        self.config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(self.model_class, self.config, **model_kwargs)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer


        if self.verbose:
            print("The total parameter required calculate "
              "gradient is:{}".format(count_parameters(self.model)))

        # 加载模型(Load Checkpoint)
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(self.model, ckpt_path)


        # This suppose not to be use in refine model.
        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )

        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

        if self.args.use_rec:
            self.critic = Critic(self.args)

        if self.args.hyperparameter_search:
            self.args.output = self.args.output + '/' + str(self.args.lr)


    def init_wandb(self):
        wandb.init(project=self.args.experiment_name)
        wandb.run.name = self.args.run_name
        wandb.config.update(self.args)

        src_dir = Path(__file__).resolve().parent
        base_path = str(src_dir.parent)
        src_dir = str(src_dir)
        wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        self.wandb_initialized = True

    def log_metric(self, loader, epoch, val_set_name='', log_str=''):
        
        assert val_set_name != '', 'please give a name to the evaluate set'
        test_results, _ = self.evaluate(loader)
        if self.verbose:
            wandb_log_dict = {}
            for score_name, score in test_results.items():
                if not (type(score) is np.ndarray):
                    wandb_log_dict[f'{val_set_name}{score_name}'] = score
            wandb.log(wandb_log_dict)

            log_str += '\n'
            test_results_for_pprint = {'CIDEr': test_results['CIDEr'],
                                    'METEOR': test_results['METEOR']}
            log_str += pformat(test_results_for_pprint)

            print(log_str)
        
    # @profile(precision=4,stream=open('memory_profiler.log','w+')) 
    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            self.best_valid = 0.
            self.best_epoch = 0

            if not self.wandb_initialized:
                self.init_wandb()


        if self.args.distributed:
            dist.barrier()

        global_step = 0

        if self.args.rl_training:
            accumulate_reward_baseline = 0
            accumulate_sample_reward = 0
            accumulate_sample_cider_reward = 0
            accumulate_reward = 0

        for epoch in range(self.args.epochs):

            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            epoch_results = {
                'loss': 0.,
                'sample_reward': 0.,
                'reward': 0.,
                'reward_baseline': 0.,
            }

            for step_i, batch in enumerate(self.train_loader):

                if self.args.distributed:
                    if self.args.rl_training:
                        if self.args.use_rec:
                            results = self.model.module.rec_rl_train_step(batch, self.critic, use_combine=self.args.use_combine, combine_with_celoss=self.args.combine_with_celoss)
                        else:
                            results = self.model.module.rl_train_step(batch)
                    elif self.args.dialog_training:
                        results = self.model.module.dialog_train_step(batch, self.critic, self.args.dialog_round)
                    else:
                        results = self.model.module.train_step(batch, self.args.use_mmi, epoch=epoch, margin=self.args.mmi_margin, use_negative_text_training=self.args.use_negative_text_training)
                else:
                    if self.args.rl_training:
                        if self.args.use_rec:
                            results = self.model.module.rec_rl_train_step(batch, self.critic, use_combine=self.args.use_combine, combine_with_celoss=self.args.combine_with_celoss)
                        else:
                            results = self.model.module.rl_train_step(batch)
                    elif self.args.dialog_training:
                        results = self.model.module.dialog_train_step(batch, self.critic, self.args.dialog_round)
                    else:
                        results = self.model.train_step(batch, self.args.use_mmi, epoch=epoch, margin=self.args.mmi_margin, use_negative_text_training=self.args.use_negative_text_training)

                loss = results['loss']
                loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad_norm)

                self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None
                global_step += 1
                
                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr
                        
                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)
                
                # for debug need
                if self.args.debug:
                    if global_step==15:
                        break
                
                if self.args.rl_training:
                    sample_reward = results['sample_reward']
                    reward = results['reward']
                    if self.args.use_combine:
                        sample_cider_reward = results['sample_cider_reward']
                        accumulate_sample_cider_reward += sample_cider_reward.item()

                    accumulate_sample_reward += sample_reward.item()
                    accumulate_reward += reward.item()

                    if not self.args.use_rec:
                        reward_baseline = results['reward_baseline']
                        accumulate_reward_baseline += reward_baseline.item()

                    if self.verbose and (global_step % 100 == 0) and global_step!=0:
                        wandb_log_dict_for_reward = {}
                        if not self.args.use_rec:
                            wandb_log_dict_for_reward['Reward/reward_baseline'] = accumulate_reward_baseline/100
                        wandb_log_dict_for_reward['Reward/sample_reward'] = accumulate_sample_reward/100
                        wandb_log_dict_for_reward['Reward/reward'] = accumulate_reward/100
                        if self.args.use_combine:
                            wandb_log_dict_for_reward['Reward/sample_cider_reward'] = accumulate_sample_cider_reward / 100
                        wandb.log(wandb_log_dict_for_reward)

                        accumulate_sample_reward = 0
                        if self.args.use_combine:
                            accumulate_sample_cider_reward = 0
                        accumulate_reward = 0
                        if not self.args.use_rec:
                            accumulate_reward_baseline = 0

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

            if self.args.distributed:
                dist.barrier()

            # 按Epoch打印训练过程的指标
            if self.verbose:
                pbar.close()
                wandb_log_dict_for_training = {}
                wandb_log_dict_for_training['Train/Loss'] = epoch_results['loss'] / len(self.train_loader)
            
                if self.args.rl_training:
                    wandb_log_dict_for_training['Train/sample_reward'] = epoch_results['sample_reward'] / len(self.train_loader)
                    wandb_log_dict_for_training['Train/reward'] = epoch_results['reward'] / len(self.train_loader)
                    if not self.args.use_rec:
                        wandb_log_dict_for_training['Train/reward_baseline'] = epoch_results['reward_baseline'] / len(self.train_loader)
                wandb.log(wandb_log_dict_for_training)
        
            # if not self.args.no_evaluate and epoch%10==0:
            if not self.args.no_evaluate:
                self.evaluate_for_one_epoch(epoch=epoch)
                
            # if self.verbose and epoch%10==0:
            if self.verbose:
                self.save(str(epoch))

            if self.args.distributed:
                dist.barrier()

        # 这里只会保存主显卡的参数权重
        if self.verbose:
            self.save("LAST")


        if self.args.distributed:
            dist.barrier()
            
    def evaluate_for_one_epoch(self, epoch):
        
        # 按验证集存储最好的ckpt
        valid_results, _ = self.evaluate(self.val_loader)
        if self.verbose:
            valid_score = valid_results['CIDEr']
            if valid_score > self.best_valid or epoch == 0:
                self.best_valid = valid_score
                self.best_epoch = epoch
                self.save("BEST")

                log_str = ''

                valid_results_for_pprint = {'CIDEr': valid_results['CIDEr'],
                                        'METEOR': valid_results['METEOR']}
                log_str += pformat(valid_results_for_pprint)
                log_str += "\nEpoch %d: Valid CIDEr %0.4f" % (epoch, valid_score)
                log_str += "\nEpoch %d: Best CIDEr %0.4f\n" % (self.best_epoch, self.best_valid)
                print(log_str)
            wandb_log_dict_for_val = {}
            for score_name, score in valid_results.items():
                if not (type(score) is np.ndarray):
                    wandb_log_dict_for_val[f'Valid/{score_name}'] = score

            wandb_log_dict_for_val[f'Valid/best_epoch'] = self.best_epoch
            wandb_log_dict_for_val['Train/epoch'] = epoch
            wandb.log(wandb_log_dict_for_val)
        
        # 记录测试集和验证集上的指标
        if self.args.dataset == 'refcocog':
            self.log_metric(loader=self.test_loaderA,epoch=epoch,val_set_name='Train/Test_', log_str='Test Result:')
        else:
            self.log_metric(loader=self.test_loaderA,epoch=epoch,val_set_name='Train/TestA_', log_str='TestA Result:')
            self.log_metric(loader=self.test_loaderB,epoch=epoch,val_set_name='Train/TestB_', log_str='TestB Result:')
    
    # @profile(precision=4,stream=open('memory_profiler.log','w+')) 
    def predict(self, loader):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        
        with torch.no_grad():

            predictions = []
            # targets = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            # gen_kwargs['num_beams'] = 5
            gen_kwargs['max_length'] = self.args.gen_max_length

            # 这块就搞不太懂...
            # for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction", disable=not self.verbose)):
            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction")):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(deepcopy(results))
            
            # if you don't have the same
            if self.args.distributed:
                dist.barrier()
                dist_results = dist_utils.all_gather(predictions)
                predictions = []
                for result in dist_results:
                    predictions.extend(result)

            return predictions
        
    def evaluate_for_ofa_acc(self, loader, generate_res):
        
        ofa_acc_sum = torch.FloatTensor([0]).cuda()
        ofa_acc_cnt = torch.FloatTensor([0]).cuda()
        ofa_score_recoder = {}
        with torch.no_grad():
        
            for i, batch in enumerate(tqdm(loader, ncols=120, desc="ofa_evaluate", disable=not self.verbose)):
                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']
                ref_ids = batch['ref_ids']
        
                sample_sents = []
                for ref_id in ref_ids:
                    sent = generate_res[ref_id]
                    sample_sents.append(sent)
        
                sample_dict['sents'] = sample_sents  # a list of sent
        
                scores, _, _ = self.critic.compute_score(sample_dict, ofa_test=True, threshold=0.5)
                ofa_acc_sum += sum(scores) if scores is not None else 0
                ofa_acc_cnt += len(scores) if scores is not None else 0
    
                for score, ref_id in zip(scores, ref_ids):
                    ofa_score_recoder[ref_id] = score.item()
        
            if self.args.distributed:
                dist.barrier()
        
                ofa_acc_sum_for_dist = dist_utils.all_gather(ofa_acc_sum.item())
                ofa_acc_cnt_for_dist = dist_utils.all_gather(ofa_acc_cnt.item())
                ofa_score_recoder_for_dist = dist_utils.all_gather(ofa_score_recoder)
                ofa_score_recoder_gather = {}
                for dict in ofa_score_recoder_for_dist:
                    ofa_score_recoder_gather.update(dict)
                ofa_acc_sum_gather = sum(ofa_acc_sum_for_dist)
                ofa_acc_cnt_gather = sum(ofa_acc_cnt_for_dist)
            else:
                ofa_acc_sum_gather = ofa_acc_sum.item()
                ofa_acc_cnt_gather = ofa_acc_cnt.item()
                ofa_score_recoder_gather = ofa_score_recoder

        ofa_results = {
            'ofa_acc_sum_gather': ofa_acc_sum_gather,
            'ofa_acc_cnt_gather': ofa_acc_cnt_gather,
            'ofa_score_recoder_gather': ofa_score_recoder_gather,
        }
        return ofa_results
    
    # @profile(precision=4,stream=open('memory_profiler.log','w+')) 
    def evaluate(self, loader, save_name=None):

        preds = self.predict(loader)
        generate_res = {}

        for pred in preds:
            generate_res[pred['ref_id']] = pred['sent']

        # In distributed mode, the distributed sampler will complement the batch size to the
        # same in every sing gpu, when the drop_last is default False. So there may be some redundant.
        if not self.args.distributed:
            assert len(preds) == len(generate_res), 'The length does not match, prediction is{}, and generate_res is{}'.format(len(preds),
                                                                                                      len(generate_res))

        if self.args.use_rec:
            ofa_results = self.evaluate_for_ofa_acc(loader=loader, generate_res=generate_res)
        
        result = {}
        if self.verbose:
            print('# predictions:', len(preds))
            evaluator = RefEvaluation(self.refer, preds)
            CIDEr_sc, CIDEr_scs, METEOR_sc, METEOR_scs = evaluator.evaluate()
            result['CIDEr'] = CIDEr_sc
            result['CIDErs'] = CIDEr_scs
            result['METEOR'] = METEOR_sc
            result['METEORs'] = METEOR_scs
            
            print("CIDEr score:{}".format(result['CIDEr']))
            print("METEOR score:{}".format(result['METEOR']))
            
            if self.args.use_rec:
                result['OFA_Acc'] = ofa_results['ofa_acc_sum_gather'] / ofa_results['ofa_acc_cnt_gather']
                print("OFA_Acc_Score:{}".format(ofa_results['ofa_acc_sum_gather'] / ofa_results['ofa_acc_cnt_gather']))
                print("OFA_Acc_Score_Sum:{}".format(ofa_results['ofa_acc_sum_gather']))
                print("OFA_Acc_Score_Cnt:{}".format(ofa_results['ofa_acc_cnt_gather']))

        return result, preds


def main_worker(gpu, args):
    
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')
    if gpu == 0:
        verbose = True
    else:
        verbose = False
    refer = REFER(dataset=args.dataset, splitBy=args.dataset_split, img_dir=args.img_dir, ref_dir=args.refcoco_dir, verbose=verbose)
    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        refer=refer,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )

    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        refer=refer,
        split=args.valid, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )
    print('# len val loader:', len(val_loader))

    print(f'Building test loader at GPU {gpu}')
    if args.dataset == 'refcocog':
        test_loader = get_loader(
            args,
            refer=refer,
            split=args.test, mode='val', batch_size=valid_batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=4,
            topk=args.valid_topk,
        )
        trainer = Trainer(args, train_loader, val_loader, test_loader, train=True, refer=refer)
    else:
        test_loaderA = get_loader(
            args,
            refer=refer,
            split='testA', mode='val', batch_size=valid_batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=4,
            topk=args.valid_topk,
        )
        test_loaderB = get_loader(
            args,
            refer=refer,
            split='testB', mode='val', batch_size=valid_batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=4,
            topk=args.valid_topk,
        )
        trainer = Trainer(args, train_loader, val_loader, test_loaderA, test_loaderB, train=True, refer=refer)

    # trainer = Trainer(args, train_loader, train=True)
    trainer.train()


if __name__ == "__main__":
    
    cudnn.benchmark = True
    args = parse_args()

    # import os
    # local_rank = int(os.environ["LOCAL_RANK"])

    refcoco_dir = Path(args.refcoco_dir)
    args.refcoco_dir = refcoco_dir.joinpath(args.dataset)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime

        current_time = datetime.now().strftime('%b%d_%H-%M')

        # run_name其实可以自己设置一下
        run_name = f'{current_time}_GPU{args.world_size}_{args.dataset}_{args.lr}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        # run_name设置在project下面的每个小实验的项目名字
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
        # main_worker(local_rank, args)
    else:
        main_worker(0, args)
