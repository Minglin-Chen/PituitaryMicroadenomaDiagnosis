import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dataset import dataset_provider
from model import model_provider

from experiment.setup import reproduce_setup, folder_setup
from experiment.engine import trainer, evaluator

# Configuration
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, required=False)
parser.add_argument('--mode', type=str, default='train', required=False)
parser.add_argument('--log_root', type=str, default='logs', required=False)
parser.add_argument('--dataset_name', type=str, default='PituitaryAdenomaCls', required=False)
parser.add_argument('--dataset_root', type=str, default='../Dataset/Diagnosis/trainval_index.npz', required=False)
parser.add_argument('--batch_size', type=int, default=16, required=False)
parser.add_argument('--model', type=str, default='baseline', required=False)
parser.add_argument('--in_channels', type=int, default=5, required=False)
parser.add_argument('--num_classes', type=int, default=2, required=False)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--loss_name', type=str, default='label_smooth', required=False)
parser.add_argument('--lr', type=float, default=0.01, required=False)
parser.add_argument('--weight_decay', type=float, default=5e-4, required=False)
parser.add_argument('--num_epoch', type=int, default=200, required=False)
parser.add_argument('--restore', type=str, default='', required=False)
parser.add_argument('--result', type=str, default='result', required=False)
args = parser.parse_args()


def train():
    ckpt_dir, tsrboard_dir = folder_setup(args)

    # 1. Load dataset
    dataset_train = dataset_provider(args.dataset_name, args.dataset_root, is_train=True)
    dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=16)
    dataset_val = dataset_provider(args.dataset_name, args.dataset_root, is_train=False)
    dataloader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=16)

    # 2. Build model
    model = model_provider(
        args.model, 
        in_channels=args.in_channels, 
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        loss_name=args.loss_name).cuda()
    model = nn.DataParallel(model)

    # 3. Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=np.exp(np.log(0.01)/args.num_epoch))

    # 4. Ops
    train_op = trainer(
        model=model,
        dataloader=dataloader_train,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=SummaryWriter(os.path.join(tsrboard_dir, 'train')),
        log_per_step=1)

    eval_train_op = evaluator(
        model=model, 
        dataloader=dataloader_train, 
        logger=SummaryWriter(os.path.join(tsrboard_dir, 'eval_train')))

    eval_val_op = evaluator(
        model=model, 
        dataloader=dataloader_val, 
        logger=SummaryWriter(os.path.join(tsrboard_dir, 'eval_val')))

    # 5. Train loop
    best_metric_dict = {}
    for epoch in range(args.num_epoch):

        # train
        print('>>> train :)')
        train_op(epoch)
        
        # evaluate
        print('>>> evaluate :)')
        eval_train_op(epoch)
        metric_dict = eval_val_op(epoch)

        # save weights
        ckpt_path = os.path.join(ckpt_dir, '{}.newest.pt'.format(args.model))
        torch.save(model.state_dict(), ckpt_path)
        if not best_metric_dict: best_metric_dict = metric_dict
        if metric_dict['main'] >= best_metric_dict['main']:
            ckpt_path = os.path.join(ckpt_dir, '{}.best.pt'.format(args.model))
            torch.save(model.state_dict(), ckpt_path)
            best_metric_dict = metric_dict
        best_metric_str = 'best metric: '
        for k, v in best_metric_dict.items():
            best_metric_str += '[{} {:.4f}] '.format(k, v)
        print(best_metric_str)


def evaluate():

    # 1. Load dataset
    dataset = dataset_provider(args.dataset_name, args.dataset_root, is_train=False)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

    # 2. Build model
    model = model_provider(
        args.model, 
        in_channels=args.in_channels, 
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        loss_name=args.loss_name).cuda()
    model = nn.DataParallel(model)
    state_dict = torch.load(args.restore)
    model.load_state_dict(state_dict)

    # 3. Op
    eval_op = evaluator(model=model, dataloader=dataloader)

    # 4. Run
    print('>>> Evaluation :)')
    eval_op()


@torch.no_grad()
def inference():
    import json

    # 1. Load dataset
    dataset = dataset_provider(args.dataset_name, args.dataset_root, is_train=False)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

    # 2. Build model
    model = model_provider(
        args.model, 
        in_channels=args.in_channels, 
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        loss_name=args.loss_name).cuda()
    model = nn.DataParallel(model)
    state_dict = torch.load(args.restore)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Run
    proba_list, label_list = [], []
    for data_dict in tqdm(dataloader):

        # 1. forward
        ret_dict = model(data_dict)
        logit = ret_dict['logit']
        proba = logit.softmax(dim=1)[:,1]
        label = data_dict['label']

        # 2. record
        proba_list += proba.tolist()
        label_list += label.tolist()
    
    # 4. Serialize
    if not os.path.exists(args.result): os.makedirs(args.result)
    with open(os.path.join(args.result, '{}_inference.json'.format(args.model)), 'w') as f:
        json.dump({'proba': proba_list, 'label': label_list}, f)


if __name__=='__main__':

    # Setup
    reproduce_setup(args.seed)
    # Run
    eval(args.mode+'()')
    
