import math
import sys
import time

from .utils import SmoothedValue
from .utils import MetricLogger
from .utils import warmup_lr_scheduler
from .utils import reduce_dict

class trainer():

    def __init__(self, model, dataloader, optimizer, scheduler, logger, log_per_step):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.log_per_step = log_per_step

    def __call__(self, epoch):
        # setup 
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        # warm-up learning rate strategy
        warmup_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.dataloader) - 1)
            warmup_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        # loop
        for i, (images, targets) in enumerate(metric_logger.log_every(self.dataloader, self.log_per_step, header)):
            # 0. place to CUDA
            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # 1. forward
            loss_dict = self.model(images, targets)

            # 2. update weights
            self.optimizer.zero_grad()
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.optimizer.step()

            if warmup_scheduler is not None:
                warmup_scheduler.step()

            # 3. reduce losses
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # 4. logger
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]['lr'])

            if self.logger is not None:
                self.logger.add_scalar('loss', loss_value, i+epoch*len(self.dataloader))
                for k, v in loss_dict_reduced.items():
                    self.logger.add_scalar(f'{k}', v, i+epoch*len(self.dataloader))
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

        if self.scheduler is not None: self.scheduler.step()