import os
import sys
import numpy as np
from cv2 import cv2
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

from experiment.setup import folder_setup, reproduce_setup
from experiment.engine import trainer, evaluator


# Configuration
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, required=False)
parser.add_argument('--mode', type=str, default='train', required=False)
parser.add_argument('--log_root', type=str, default='logs', required=False)
parser.add_argument('--dataset_name', type=str, default='PituitaryAdenomaDet', required=False)
parser.add_argument('--dataset_root', type=str, default='../Dataset/Detection/trainval_index.npz', required=False)
parser.add_argument('--batch_size', type=int, default=32, required=False)
parser.add_argument('--model', type=str, default='FasterRCNN', required=False)
parser.add_argument('--num_classes', type=int, default=2, required=False)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--lr', type=float, default=5e-3, required=False)
parser.add_argument('--weight_decay', type=float, default=5e-4, required=False)
parser.add_argument('--num_epoch', type=int, default=20, required=False)
parser.add_argument('--restore', type=str, default='', required=False)
parser.add_argument('--result', type=str, default='result', required=False)
args = parser.parse_args()


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    ckpt_dir, tsrboard_dir = folder_setup(args)

    # 1. Load dataset
    dataset_train = dataset_provider(args.dataset_name, args.dataset_root, is_train=True)
    dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    dataset_val = dataset_provider(args.dataset_name, args.dataset_root, is_train=False)
    dataloader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 2. Build model
    model = model_provider(args.model, num_classes=args.num_classes, pretrained=args.pretrained).cuda()
    # model = nn.DataParallel(model)

    # 3. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. Ops
    train_op = trainer(
        model=model,
        dataloader=dataloader_train,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=SummaryWriter(os.path.join(tsrboard_dir, 'train')),
        log_per_step=10)

    eval_train_op = evaluator(
        model=model, 
        dataloader=dataloader_train, 
        logger=SummaryWriter(os.path.join(tsrboard_dir, 'eval_train')))

    eval_val_op = evaluator(
        model=model, 
        dataloader=dataloader_val, 
        logger=SummaryWriter(os.path.join(tsrboard_dir, 'eval_val')))

    # 5. Train loop
    for epoch in range(args.num_epoch):

        # train
        print('>>> train :)')
        train_op(epoch)
        
        # evaluate
        print('>>> evaluate :)')
        eval_train_op(epoch)
        eval_val_op(epoch)

        # save weights
        ckpt_path = os.path.join(ckpt_dir, '{}.pt'.format(args.model))
        torch.save(model.state_dict(), ckpt_path)


def evaluate():

    # 1. Load dataset
    dataset = dataset_provider(args.dataset_name, args.dataset_root, is_train=False)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Build model
    model = model_provider(args.model, num_classes=args.num_classes, pretrained=args.pretrained).cuda()
    state_dict = torch.load(args.restore)
    model.load_state_dict(state_dict)

    # 3. Op
    eval_op = evaluator(model=model, dataloader=dataloader)

    # 4. Run
    print('>>> Evaluation :)')
    eval_op()


def visualize():

    # 1. Load dataset
    dataset = dataset_provider(args.dataset_name, args.dataset_root, is_train=False)
    dataloader = DataLoader(dataset, 2, shuffle=False, collate_fn=collate_fn)

    # 2. Build model
    model = model_provider(args.model, num_classes=args.num_classes, pretrained=args.pretrained).cuda()
    state_dict = torch.load(args.restore)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Loop
    if not os.path.exists(args.result): os.makedirs(args.result)
    for i, (images, targets) in enumerate(tqdm(dataloader)):
        # forward
        det_res = [model([image.cuda()])[0] for image in images]
        det_res = [{k:v.cpu() for k,v in res.items()} for res in det_res]
        targets = [{k:v.cpu() for k,v in t.items()} for t in targets]
        # show
        for j, (image, tgt, res) in enumerate(zip(images, targets, det_res)):
            # convert to rgb
            image_rgb = (image[0].numpy() * 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
            # draw box - traget
            for box in tgt['boxes']:
                cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (200,0,200), 3)
            # draw box - pred
            for score, box in zip(res['scores'], res['boxes']):
                cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (0,200,0), 2)
                cv2.putText(image_rgb, '{:.4f}'.format(score.item()), (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
            
            cv2.imwrite(f'{args.result}/{i:04}_{j:04}.png', image_rgb)
            # cv2.imshow('result', image_rgb)
            # cv2.waitKey()


if __name__=='__main__':

    # Setup
    reproduce_setup(args.seed)
    # Run
    eval(args.mode+'()')