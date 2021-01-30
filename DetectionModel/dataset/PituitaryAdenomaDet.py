import os
import numpy as np
from cv2 import cv2
from PIL import Image

import torch
import torchvision.transforms as T
from albumentations import HorizontalFlip, VerticalFlip, Compose, BboxParams


class PituitaryAdenomaDet():
    r"""
    Dataset for pituitary adenoma detection (2D)
    Args:
        index_path: str, path to file
        split: str, 'train' or 'val'
        image_size: tuple, specify the size of image
    """
    def __init__(self, index_path, split, image_size=(256, 256)):
        self.examples = np.load(index_path, allow_pickle=True)[split]
        self.root = os.path.dirname(index_path)
        self.split = split
        self.image_size = image_size
        self.transform = T.Compose([
            T.ToTensor()])
        self.al_transform = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5)
        ], bbox_params=BboxParams(format='pascal_voc', label_fields=['category_id']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # get examples
        path, rect, _ = self.examples[index]
        path = os.path.join(self.root, path).replace('\\', '/')

        # load image
        image = cv2.imread(path)

        # refine rectangle
        h, w, _ = image.shape
        hn, wn = self.image_size

        # resize
        image_n = cv2.resize(image, dsize=self.image_size)
        center_hn, center_wn = (rect[3]+rect[1])/2.0/h*hn, (rect[2]+rect[0])/2.0/w*wn
        rect_hn_refine, rect_wn_refine = 0.1 * hn, 0.2 * wn
        boxes = [[
            int(center_wn-rect_wn_refine/2),
            int(center_hn-rect_hn_refine/2),
            int(center_wn+rect_wn_refine/2),
            int(center_hn+rect_hn_refine/2)]]
        labels = [1]

        # paired augmentation
        # if self.split == 'train':
        #     augmentated = self.al_transform(image=image_n, bboxes=boxes, category_id=labels)
        #     image_n, boxes, labels = augmentated['image'], augmentated['bboxes'], augmentated['category_id']

        # convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # target dict
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        image_n = Image.fromarray(image_n)
        image_n = self.transform(image_n)

        return image_n, target


def _visulize_dataset():    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default='Dataset/Detection/trainval_index.npz', required=False)
    parser.add_argument('--split', type=str, default='train', required=False)
    args = parser.parse_args()

    dataset = PituitaryAdenomaDet(args.index_path, args.split)
    patients = [ os.path.basename(os.path.dirname(path)) for path, _, _ in dataset.examples]
    print('#patient {}, #slice {}'.format(len(set(patients)), len(dataset)))

    for image, target in dataset:
        image_np = image.permute([1,2,0]).numpy()
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        box = target['boxes'][0]
        image_render = cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0,0,255), 3)

        cv2.imshow('image', image_render)
        cv2.waitKey(10)


if __name__=='__main__':

    _visulize_dataset()