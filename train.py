import torch
import argparse
import os
import random
import numpy as np
from torch.backends import cudnn
from model import anchor_model
from config.config import cfg, cfg_from_file, cfg_from_list
from tools.prepare_data import prepare_data_Anchor
import pprint


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--method', dest='method',
                        help='set the method to use',
                        default='Anchor', type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name',
                        default='exp', type=str)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def setup_random_seed(seed):
    print(f"\nSet random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    bn_domain_map = {}

    if args.method == 'Anchor':
        from solver.anchor_solver import AnchorSolver as Solver
        dataloaders = prepare_data_Anchor()
        num_domains_bn = 1

    else:
        raise NotImplementedError("Currently don't support the specified method: %s."
                                  % args.method)

    # initialize model
    feature_extractor_state_dict = torch.load(cfg.MODEL.PRETRAINED_FEATURE_EXTRACTOR, map_location='cpu')

    net = anchor_model.anchornet(num_classes=cfg.DATASET.NUM_CLASSES,
                                 feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR,
                                 feature_extractor_state_dict=feature_extractor_state_dict,
                                 frozen=[cfg.TRAIN.STOP_GRAD],
                                 dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                                 fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                                 num_domains_bn=num_domains_bn)

    net = torch.nn.DataParallel(net.cuda())

    train_solver = Solver(net, dataloaders, bn_domain_map=bn_domain_map)
    train_solver.solve()
    print('Finished!')


if __name__ == '__main__':
    cudnn.benchmark = True

    args = parse_args()

    print('Called with args:')
    print(args)
    setup_random_seed(args.seed)

    assert (args.cfg_file is not None) and (args.exp_name is not None)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.resume is not None:
        cfg.RESUME = args.resume
    if args.weights is not None:
        cfg.MODEL = args.weights

    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name

    cfg.EXP_NAME = os.path.split(args.cfg_file)[-1].split('cfg.yaml')[0] + args.exp_name
    if 'visda' in cfg.EXP_NAME:
        cfg.EXP_NAME = cfg.EXP_NAME.replace('_train2val', '')
    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    assert cfg.MODEL.PRETRAINED_FEATURE_EXTRACTOR != ''
    if '{}' in cfg.MODEL.PRETRAINED_FEATURE_EXTRACTOR:
        cfg.MODEL.PRETRAINED_FEATURE_EXTRACTOR = \
            cfg.MODEL.PRETRAINED_FEATURE_EXTRACTOR.format(cfg.DATASET.SOURCE_NAME,
                                                          cfg.DATASET.TARGET_NAME)

    print('Using config:')
    pprint.pprint(cfg)

    train(args)
