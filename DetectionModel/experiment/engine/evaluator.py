import time

import torch
import torchvision

from .utils import MetricLogger

from .coco_eval import CocoEvaluator, get_coco_api_from_dataset

class evaluator():

    def __init__(self, model, dataloader, logger=None):
        self.model = model
        self.dataloader = dataloader
        self.logger = logger

    @torch.no_grad()
    def __call__(self, epoch=0):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)

        # setup
        self.model.eval()

        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(self.dataloader.dataset)
        iou_types = ['bbox'] # self._get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        # loop
        for image, targets in metric_logger.log_every(self.dataloader, 10, header):
            # 0. place to CUDA
            image = list(img.cuda() for img in image)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # 1. forward
            torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(image)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            # 2. metrics
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            
            # 3. logger
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        # accumulate predictions from all images
        # coco_evaluator.coco_eval['bbox'].params.maxDets[2] = 100
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)

        # logger
        if self.logger is not None:
            self.logger.add_scalar('AP@0.50', coco_evaluator.coco_eval['bbox'].stats[1], epoch)
            self.logger.add_scalar('AP@0.75', coco_evaluator.coco_eval['bbox'].stats[2], epoch)

        return coco_evaluator

    def _get_iou_types(self, model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types