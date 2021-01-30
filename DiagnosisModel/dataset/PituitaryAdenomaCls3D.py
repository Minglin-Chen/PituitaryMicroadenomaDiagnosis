import os
import numpy as np
from cv2 import cv2

from albumentations import Compose, GaussNoise, ShiftScaleRotate


class PituitaryAdenomaCls3D():
    r"""
    Dataset for pituitary adenoma classification (3D)
    Args:
        index_path: str, path to file
        split: str, 'train' or 'val'
    """
    def __init__(self, index_path, split):
        self.root = os.path.dirname(index_path)
        self.split = split + '_patient'
        self.examples = np.load(index_path, allow_pickle=True)[self.split]
        self.is_train = True if split == 'train' else False
        self.aug_patch = Compose([
            GaussNoise(var_limit=20, p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=0)])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        r"""
        Return:
            dict which has the following keys and values:
                'input': numpy.ndarray (1,5,H,W)
                'label': int32
        """
        # get example
        patient_example = self.examples[index]
        # patch collection
        patches = []
        for path, rect, target in patient_example:
            # load
            path = os.path.join(self.root, path).replace('\\', '/')
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, dsize=(256,256))
            # crop + resize + normalization
            patch = image[rect[1]:rect[3], rect[0]:rect[2]]
            patch = cv2.resize(patch, dsize=(256,256))
            # patch augmentation
            if self.is_train: patch = self.aug_patch(image=patch)['image']
            patch = (patch - 128.0) / 128.0
            # collect
            patches.append(patch)
        patches = np.stack(patches, axis=0)
        
        data_dict = {
            'input': patches[None].astype(np.float32), 
            'label': target}
        return data_dict


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default='Dataset/Diagnosis/trainval_index.npz', required=False)
    parser.add_argument('--split', type=str, default='train', required=False)
    args = parser.parse_args()

    dataset = PituitaryAdenomaCls3D(args.index_path, args.split)
    print(len(dataset))

    for data_dict in dataset:
        # unpack
        patches, label = data_dict['input'], data_dict['label']
        # info
        print('label [type {}] [value {}]'.format(type(label), label))
        print('patches [type {}] [dtype {}] [shape {}] [(min: {:.4f}, max: {:.4f})]'.format(
            type(patches), patches.dtype, patches.shape, patches.min(), patches.max()))
        # visulization
        for i, patch in enumerate(patches[0]):
            patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow('patch-{}'.format(i), patch)
        cv2.waitKey()
