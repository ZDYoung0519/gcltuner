import argparse
import json
import logging
import os
import os.path as osp
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from timm.utils.metrics import accuracy

import math
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dist import all_gather, is_main_process, get_rank, get_world_size, broadcast_object_list
from mmengine.device import get_device

import joblib

from sklearn.cluster import KMeans
from tqdm import tqdm
from projects.aligncl.utils.metric import MetricLogger, SmoothedValue
from sklearn.mixture import GaussianMixture


def safe_accuracy(logits, targets):
    print(logits.size())
    max_topk = min(5, logits.shape[1])
    topk = (1, max_topk)
    return accuracy(logits, targets, topk=topk)


def save_gmm(gmm, path):
    joblib.dump(gmm, path)

def load_gmm(path):
    gmm = joblib.load(path)
    return gmm


class ContinualGMMClassifier(object):
    def __init__(self):
        self.gmms = {}
    
    def update_task(self, X, task_id:int):
        gmm = GaussianMixture(
            10, 
            covariance_type='diag',
            init_params='kmeans',
            verbose=True
            # n_init=3
        )
        gmm.fit(X)
        self.gmms[task_id] = gmm
        return gmm

    def predict_task_scores(self, x):
        logits = np.array([
            gmm.score_samples(x) for gmm in self.gmms.values()
        ])
        return logits.transpose()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a router')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--saved-feature-dir', required=True)
    parser.add_argument('--cur-task', type=int, default=0, required=True)
    parser.add_argument('--num_clusters_per_task', type=int, default=10, help='the dir to save logs and models')
    parser.add_argument('--cov_mode', type=str, default='diag', help='the dir to save logs and models')
    parser.add_argument("--l2-norm", action='store_true')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def load_clip_features(saved_feature_dir, task_id, split):
    task_feature_dir = osp.join(saved_feature_dir, f"task{task_id}")
    vision_path = osp.join(task_feature_dir, f"vision_embedding_{split}.pt")
    text_path = osp.join(task_feature_dir, f"text_embeddings_{split}.pt")
    if osp.exists(vision_path) and osp.exists(text_path):
        vision_features = torch.load(vision_path, map_location="cpu")
        text_features = torch.load(text_path, map_location="cpu")
        return torch.cat([vision_features, text_features], dim=-1)

    legacy_feature_path = osp.join(saved_feature_dir, "train_features.pth", f"task{task_id}.pt")
    if osp.exists(legacy_feature_path):
        print_log(f"Load legacy {split} features from: {legacy_feature_path}")
        return torch.load(legacy_feature_path, map_location="cpu")

    raise FileNotFoundError(
        f"Can not find precomputed {split} features for task{task_id}. "
        f"Expected {vision_path} and {text_path}."
    )


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # customize
    cfg.model.cur_task = args.cur_task
    # set work dir and exp name
    cfg.visualizer = None


    gmms_classifier = ContinualGMMClassifier()

    # load previous distritbuion
    import shutil

    if args.cur_task > 0:
        pre_ckpt_dir = args.work_dir.replace(f"/task{args.cur_task}", f"/task{args.cur_task-1}")
        for j in range(args.cur_task):    
            src = osp.join(pre_ckpt_dir, f'gmm{j}.joblib')
            dst = osp.join(args.work_dir, f'gmm{j}.joblib')
            shutil.copy(src, dst)
            print("Copy from {} to {}".format(src, dst))

            gmms_classifier.gmms[j] = load_gmm(dst)


    # fit cur_task train distribution
    task_features = load_clip_features(args.saved_feature_dir, args.cur_task, "train")
    task_features = task_features.detach().cpu().numpy().astype(np.float64)
    print(task_features.shape, task_features.dtype, type(task_features))
    gmm = gmms_classifier.update_task(task_features, args.cur_task)
    save_gmm(gmm, osp.join(args.work_dir, f'gmm{args.cur_task}.joblib'))

    def validate_until_now(gmms_classifier, cur_task):
        val_acc1, val_acc5 = 0, 0
        device = get_device()
        for task_id in range(cur_task+1):
            task_features = load_clip_features(args.saved_feature_dir, task_id, "test").detach().cpu().numpy()

            logits = gmms_classifier.predict_task_scores(task_features)
            logits[:, (cur_task+1):] = float('-inf')

            targets = torch.LongTensor([task_id] * logits.shape[0]).to(device)
            logits = torch.from_numpy(logits).to(device)
            acc1, acc5 = safe_accuracy(logits, targets)

            print(f"Valid on task {task_id}, acc1: {acc1}, acc5: {acc5}")
            val_acc1 += acc1
            val_acc5 += acc5
        val_acc1 /= cur_task+1
        val_acc5 /= cur_task+1
        return val_acc1, val_acc5

    val_acc1, val_acc5 = validate_until_now(gmms_classifier, args.cur_task)
    print("Average: acc1: {}, acc5: {}".format(val_acc1, val_acc5))



if __name__ == '__main__':
    main()
