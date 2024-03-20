import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    parser.add_argument('--submit', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--refine_load', type=str, default=None, help='Load the refine model')
    parser.add_argument('--base_load', type=str, default=None, help='Load the base model, actually for refiner test task!')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default=None)

    parser.add_argument('--feat_dim', type=float, default=2048)
    parser.add_argument('--pos_dim', type=float, default=4)

    parser.add_argument('--use_vision', default=True, type=str2bool)
    parser.add_argument('--use_vis_order_embedding', default=True, type=str2bool)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--individual_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--losses", default='lm,obj,attr,feat', type=str)

    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=20)

    # Data
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--n_boxes', type=int, default=36)
    parser.add_argument('--max_n_boxes', type=int, default=36)
    parser.add_argument('--max_text_length', type=int, default=20)
    parser.add_argument("--classifier", action='store_true')

    parser.add_argument("--shuffle_boxes", action='store_true')
    parser.add_argument('--dataset', type=str, default='refcoco')
    parser.add_argument('--dataset_split', type=str, default='unc')
    parser.add_argument('--split', type=str, default='testA')
    parser.add_argument('--refcoco_dir', type=str, default=None)
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--ofa_ckpt_dir', type=str, default=None)
    parser.add_argument('--bad_re_save_path', type=str, default=None)
    
    # MMI training
    parser.add_argument('--use_mmi', action='store_true')
    parser.add_argument('--lama', type=float, default=1)
    parser.add_argument('--mmi_margin', type=float, default=0.5)

    # RL training
    parser.add_argument('--rl_training', action='store_true')

    # Multitask
    parser.add_argument("--multitask_sampling", type=str, default='roundrobin')
    parser.add_argument("--tasks", type=str, default='')

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_evaluate', action='store_true')
    parser.add_argument('--use_rec', action='store_true')
    parser.add_argument('--hyperparameter_search', action='store_true')
    parser.add_argument('--ofa_test', default=False, type=bool)
    parser.add_argument("--use_combine", action='store_true')
    parser.add_argument("--combine_with_celoss", action='store_true')
    parser.add_argument("--mode", type=str, default='train')

    # dialog
    parser.add_argument("--dialog_training", action='store_true')
    parser.add_argument("--dialog_round", type=int, default=1)
    parser.add_argument("--zero_shot_test", action='store_true')
    parser.add_argument("--last_round", action='store_true')
    parser.add_argument("--dialog_sp_training", action='store_true')
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--test_threshold", type=float, default=0.5, help="REC threshold in test time")
    parser.add_argument("--bad_res_path", type=str, default=None, help="the path of bad referring expression collected by REC")
    parser.add_argument("--use_detector", action='store_true')
    parser.add_argument("--only_dialog_loss", action="store_true")

    # bootstrap training
    parser.add_argument("--use_bootstrap", action='store_true')
    parser.add_argument("--bootstrap_data", type=str, default=None, help="assign the path of bootstrap data")
    parser.add_argument("--use_negative_text_training", action='store_true')
    parser.add_argument("--negative_text_training_data", type=str, default=None, help="assign the path of bootstrap data")
    
    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
