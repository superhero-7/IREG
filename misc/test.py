import sys
sys.path.insert(0, '/data/codebase/ireg/src')

import json
from pathlib import Path
from copy import deepcopy
from reg import Trainer
from tools.param import parse_args
from reg_data import get_loader
from reg_data import RefCOCOGenerationFineTuneDataset
from eval_utils.refcoco_utils import REFER
from loguru import logger


### 这个函数需要做到的功能就跟训练中，每个epoch的测试是一样的，所以我们只需要把Trainer中的写好，在这个地方一样地做就行了； ###

def test(ckpt_path, dataset, split, save_path=None, gpu=0):

    logger.add('./eval.log')
    logger.info(f'Ckpt is {ckpt_path}, Dataset is {dataset}, split is {split}, Results Saved in {save_path}')
    
    args = parse_args()
    args.gpu = gpu
    args.train = 'val'
    args.num_beams = 5
    args.batch_size = 1
    args.dataset = dataset
    split_map = {'refcoco+': 'unc',
                 'refcoco': 'unc',
                 'refcocog': 'umd'}
    args.dataset_split = split_map[args.dataset]
    args.load= ckpt_path
    args.rl_training = False
    args.use_rec = True
    args.experiment_name = '2022.11.09'
    args.dialog_training = False
    args.dialog_round = 5
    args.zero_shot_test = False
    args.last_round = True
    # args.use_detector = True
    # args.refine = False
    args.test_threshold = 0.5
    args.dialog_sp_training = False
    # args.refine_load = '/raid_sda/yfl/codebase/VL-T5-REG/VL-T5/snap/' + args.dataset + '/' + \
    #                    'vlt5_ofa_mmi_dialog_sp_training_threshold_0.5_use_region_feature' + '/' + '5e-05' + '/' + "LAST"
    # args.bad_res_path = './REG_mmi_refcocog_vlt5_bad_sent_threshold_0.5_with_bbox.json'
    args.mode = 'val'
    args.distributed = False
    args.refcoco_dir = '/data/database/REGDATA/RefCOCO'
    args.img_dir = '/data/database/REGDATA/train2014'
    args.ofa_ckpt_dir = '/data/database/REGDATA/ofa_ckpt/'
    
    refcoco_dir = Path(args.refcoco_dir)
    args.refcoco_dir = refcoco_dir.joinpath(args.dataset)

    # val_loader = get_loader(
    #     args,
    #     split=split, mode='val', batch_size=args.batch_size,
    #     distributed=args.distributed, gpu=args.gpu,
    #     workers=args.num_workers,
    #     topk=args.train_topk,
    # )
    refer = REFER(args.dataset, args.dataset_split, img_dir=args.img_dir, ref_dir=args.refcoco_dir, verbose=True)
    val_loader = get_loader(
    args,
    refer=refer,
    split=split, mode='val', batch_size=args.batch_size,
    distributed=False, gpu=args.gpu,
    workers=4,
    topk=args.valid_topk,
    )

    # 下面这两行奇怪地触发了OFA acc的bug...我并不知道是为什么！
    # args_train = deepcopy(args)
    # args_train.dialog_sp_training = True
    trainer = Trainer(args, train=False, refer=refer)

    Score, results = trainer.evaluate(val_loader)

    for k,v in Score.items():
        if type(v) is not list:
            logger.info(f"{k} : {v}")
    
    # print(len(Score['CIDErs']))
    if save_path:
        i = 0
        for item in results:
            item['cider'] = Score['CIDErs'][i]
            item['meteor'] = Score['METEORs'][i]
            i = i+1
    
        data = json.dumps(results)
        with open(save_path, 'w') as f:
            f.write(data)


if __name__ == '__main__':
    
    import sys
    
    # 我服了，我这辈子都不想再用argpaser了...不想折腾了...就在这里换不同的ckpt_path就好了;

    ckpt_path = '/data/database/IREG_ckpt_save/refcocog_ckpt/vlt5_ofa_scst_combine_clamp_mmi/5e-06/1'
    dataset = 'refcocog'
    gpu = 0
    if dataset in ['refcoco', 'refcoco+']:
        test(ckpt_path, dataset, split='testA', save_path=None, gpu=gpu)
        test(ckpt_path, dataset, split='testB', save_path=None, gpu=gpu)
    else:
        test(ckpt_path, dataset, split='val', save_path=None, gpu=gpu)