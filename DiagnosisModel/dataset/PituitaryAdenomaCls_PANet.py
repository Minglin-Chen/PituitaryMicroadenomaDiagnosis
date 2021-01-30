import os
import numpy as np
from cv2 import cv2

from albumentations import Compose, GaussNoise, ShiftScaleRotate


def histogram_matching(image_src, image_tgt=None, hist_tgt=None):
    r"""
    Args:
        image_src: np.array
        image_tgt: np.array
        hist_tgt: np.array
    """
    hist_src = np.bincount(np.ravel(image_src), minlength=256) / np.prod(image_src.shape)
    if image_tgt is not None:
        hist_tgt = np.bincount(np.ravel(image_tgt), minlength=256) / np.prod(image_tgt.shape)

    s_size = 2048
    # r -> s -> z
    # 1. r -> s
    r2s = np.add.accumulate(hist_src) * (s_size-1)
    r2s = r2s.astype(np.int32).tolist()
    # 2. s -> z
    z2s = np.add.accumulate(hist_tgt) * (s_size-1)
    z2s = z2s.astype(np.int32).tolist()
    assert len(r2s)==256 and len(z2s)==256
    s2z = [[] for _ in range(s_size)]
    for z, s in enumerate(z2s): 
        s2z[s].append(z)
    prev_v = 0
    for i in range(len(s2z)):
        items = s2z[i]
        if len(items) == 0:
            s2z[i] = prev_v
        else:
            v = items[0]
            s2z[i] = v
            prev_v = v

    # 3. r -> z
    r2z = [0] * 256
    for r in range(256):
        s = r2s[r]
        z = s2z[s]
        r2z[r] = z

    # mapping
    image_res = np.zeros_like(image_src)
    for i in range(256): image_res[image_src==i] = r2z[i]

    return image_res


class PituitaryAdenomaCls_PANet():
    r"""
    Dataset for pituitary adenoma classification (2D)
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
                'input': numpy.ndarray (5,H,W)
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
            # collect
            patches.append(patch)
        patches = np.stack(patches, axis=0)

        # intensity shift data augmentation
        if self.is_train:
            pos = np.random.randint(30)
            patches = patches.astype(np.float32) + pos
            patches[patches>255] = 255
            patches = patches.astype(np.uint8)
        
        # histogram matching normalization
        for i in [0,1,3,4]:
            patches[i] = histogram_matching(patches[i], patches[2])

        patches = (patches - 128.) / 128.
        data_dict = {
            'input': patches.astype(np.float32), 
            'label': target}
        return data_dict


def _stat_histogram():
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default='Dataset/Diagnosis/trainval_index.npz', required=False)
    parser.add_argument('--split', type=str, default='train', required=False)
    args = parser.parse_args()

    # statistic
    root = os.path.dirname(args.index_path)
    examples = np.load(args.index_path, allow_pickle=True)[f'{args.split}_patient']
    dataset_hist = np.zeros((5,256))
    for patient_example in tqdm(examples):
        for i, (path, rect, target) in enumerate(patient_example):
            # load
            path = os.path.join(root, path).replace('\\', '/')
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, dsize=(256,256))
            # statistic
            slice_hist = np.bincount(np.ravel(image), minlength=256) / np.prod(image.shape)
            dataset_hist[i] += slice_hist
    dataset_hist = dataset_hist / len(examples)
    # print
    # print(dataset_hist)
    print(dataset_hist.sum())
    # show
    plt.figure()
    for i in range(i):
        plt.plot(dataset_hist[i])
    plt.show()
    

def _visulize_dataset():    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default='Dataset/Diagnosis/trainval_index.npz', required=False)
    parser.add_argument('--split', type=str, default='val', required=False)
    args = parser.parse_args()

    dataset = PituitaryAdenomaCls_PANet(args.index_path, args.split)
    print(len(dataset))

    for data_dict in dataset:
        # unpack
        patches, label = data_dict['input'], data_dict['label']
        # info
        print('label [type {}] [value {}]'.format(type(label), label))
        print('patches [type {}] [dtype {}] [shape {}] [(min: {:.4f}, max: {:.4f})]'.format(
            type(patches), patches.dtype, patches.shape, patches.min(), patches.max()))
        # visulization
        for i, patch in enumerate(patches):
            patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow('patch-{}'.format(i), patch)
        cv2.waitKey()


if __name__=='__main__':

    # _stat_histogram()
    _visulize_dataset()
