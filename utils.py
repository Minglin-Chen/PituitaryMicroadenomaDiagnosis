import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cv2 import cv2
import SimpleITK as sitk

import torch


def get_hist_peak_position(image, alpha=0.95):

    hist = np.bincount(np.ravel(image), minlength=256) / np.prod(image.shape)
    
    start_flag, duration, pos, max_value = False, 0, -1, -1
    for i in range(256):
        current_ind = 255-i
        current_value = hist[current_ind]
        if current_value > 0.01 and not start_flag:
            start_flag, duration, pos, max_value = True, 1, current_ind, current_value
        if current_value < 0.01 and duration > 10:
            break
        if start_flag:
            duration += 1
            if current_value >= max_value:
                pos, max_value = current_ind, current_value
    assert pos != -1
    return pos, max_value


def histogram_matching(image_src, image_tgt=None, hist_tgt=None):
    r"""
    Args:
        image_src: 
        image_tgt:
        hist_tgt:
    """
    hist_src = np.bincount(np.ravel(image_src), minlength=256) / np.prod(image_src.shape)
    if image_tgt is not None:
        hist_tgt = np.bincount(np.ravel(image_tgt), minlength=256) / np.prod(image_tgt.shape)

    s_size = 2048 # 256
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
    # prev_v = 0
    # for i in range(len(s2z)):
    #     items = s2z[i]
    #     if len(items) == 0:
    #         s2z[i] = prev_v
    #     else:
    #         v = int(np.mean(items))
    #         s2z[i] = v
    #         prev_v = v
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


def load_dicom_series(directory, series_description='T1W_SE+C'):
    r"""
    Load DICOM series with specific series description.
    Refer to:
        `https://simpleitk.org/doxygen/latest/html/Python_2DicomSeriesReader2_8py-example.html`
        `https://simpleitk.org/SPIE2019_COURSE/02_images_and_resampling.html`
        `http://dicom.nema.org/medical/dicom/current/output/pdf/part06.pdf`

    Args:
        directory: str, root to the DICOM series
        series_description: str, 'T1W_SE+C' or 'T1W_TSE_Dyn'
    Return:
        None or numpy.ndarray
    """
    reader = sitk.ImageSeriesReader()
    # setup for additional information loading
    reader.MetaDataDictionaryArrayUpdateOn()
    series = reader.GetGDCMSeriesIDs(directory)
    # find the specific madality serie
    image_np = None
    for serie in series:
        dicom_names = reader.GetGDCMSeriesFileNames(directory, serie)
        if len(dicom_names):
            # load
            reader.SetFileNames(dicom_names)
            image_sitk = reader.Execute()

            # tag for Series Description: '0008|103E'
            if series_description not in reader.GetMetaData(0, '0008|103e'): continue

            # convert to numpy.ndarray
            image_np = sitk.GetArrayFromImage(image_sitk) # z, y, x
            image_np = image_np.astype(np.float32)

            # clip the intensity
            # tag for Window Center: '(0028,1050)'
            window_center = float(reader.GetMetaData(0, '0028|1050'))
            # tag for Window Width: '(0028,1051)'
            window_width = float(reader.GetMetaData(0, '0028|1051'))
            window_lower = window_center - window_width / 2.0
            window_upper = window_center + window_width / 2.0 

            image_np = np.clip(image_np, window_lower, window_upper)
            image_np = (image_np - window_lower) / (window_upper - window_lower) * 255
            image_np = image_np.astype(np.uint8)

    for i in [0,1,3,4]:
        image_np[i] = histogram_matching(image_np[i], image_np[2])
    image_np = image_np.astype(np.float32)

    return image_np


def print_dicom_series_description(directory):
    reader = sitk.ImageSeriesReader()
    # setup for additional information loading
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    series = reader.GetGDCMSeriesIDs(directory)
    # find the specific madality serie
    for serie in series:
        dicom_names = reader.GetGDCMSeriesFileNames(directory, serie)
        if len(dicom_names):
            # load
            reader.SetFileNames(dicom_names)
            reader.Execute()
            # tag for Series Description: '0008|103E'
            print('Series Description: ', reader.GetMetaData(0, '0008|103e'))
            # tag for Window Center: '(0028,1050)'
            print('Window Center: ', float(reader.GetMetaData(0, '0028|1050')))
            # tag for Window Width: '(0028,1051)'
            print('Window Width: ', float(reader.GetMetaData(0, '0028|1051')))


def load_jpg_slices(directory):
    r"""
    Args:
        directory: str
    Returns:
        numpy.ndarray (B,C,H,W) with range [0,255]
    """
    # search path
    slice_paths = sorted(glob(os.path.join(directory, '*.jpg')))
    assert len(slice_paths) == 5
    # proper size
    slices_np = []
    for i, path in enumerate(slice_paths):
        # load
        slice_image = cv2.imread(path)
        # adjust size
        h, w, _ = slice_image.shape
        if w >= h:
            slice_image = cv2.resize(slice_image, (int(256/h*w), 256))
            w_start = slice_image.shape[1]//2 - 256//2
            w_end = w_start + 256
            slice_image = slice_image[:,w_start:w_end]
        else:
            slice_image = cv2.resize(slice_image, (256, int(256/w*h)))
            h_start = slice_image.shape[0]//2 - 256//2
            h_end = h_start + 256
            slice_image = slice_image[h_start:h_end,:]
        slice_image = slice_image.transpose(2,0,1)
        slices_np.append(slice_image)
    slices_np = np.stack(slices_np)

    for i in [0,1,3,4]:
        slices_np[i] = histogram_matching(slices_np[i], slices_np[2])
        
    slices_np = slices_np.astype(np.float32)

    return slices_np


def preprocessing_on_dicom(image_np):
    r"""
    Reisze to 256x256
    Args:
        image_np: numpy.ndarray (Z,X,Y)
    Return:
        image_np: numpy.ndarray (B,C,H,W)
    """
    # resize
    assert image_np.shape[1] == image_np.shape[2]
    slices = []
    for slice_np in image_np:
        slice_np = cv2.resize(slice_np, (256,256))
        slice_np = cv2.cvtColor(slice_np, cv2.COLOR_GRAY2BGR)
        slice_np = slice_np.transpose((2,0,1))
        slices.append(slice_np)
    image_np = np.stack(slices)
    image_np = image_np / 256
    
    return image_np


def preprocessing_on_slice(slices_np):
    r"""
    Normalization in [0,1)
    Args:
        slices_np: numpy.ndarray (B,C,H,W)
    Return:
        slices_np: numpy.ndarray (B,C,H,W)
    """
    # normalize
    slices_np = slices_np / 256

    return slices_np


def topk_boxes_from_detection(images, det_res, k=5):
    r"""
    Pick up top-K slices.
    Args:
        images: numpy.ndarray (B,C,H,W)
        det_res: List[dict] and dict should have the following keys and values:
            'scores': torch.FloatTensor (N)
            'boxes': torch.FloatTensor (N,4)
    Returns:
        images_topk: torch.FloatTensor (k,C,H,W)
        boxes_topk: torch.FloatTensor (k,4)
    """
    # select only one box with the highest score in each image
    images_good, scores_good, boxes_good = [], [], []
    for image, res in zip(images, det_res):
        scores, boxes = res['scores'], res['boxes']
        # nothing in this image
        if scores.shape[0] == 0: continue
        # highest score
        ind_good = torch.argmax(scores)
        score_good, box_good = scores[ind_good], boxes[ind_good]
        # record
        images_good.append(image)
        scores_good.append(score_good)
        boxes_good.append(box_good)
    assert len(scores_good) != 0
    images_good = np.stack(images_good)
    scores_good = torch.stack(scores_good)
    boxes_good = torch.stack(boxes_good)
    # select the top-k boxes
    ind = torch.argsort(scores_good, descending=True)
    if ind.shape[0] >= k:
        ind_topk = ind[:k]
    else:
        raise NotImplementedError
    ind_topk = torch.sort(ind_topk, descending=False)[0]
    images_topk = images_good[ind_topk]
    boxes_topk = boxes_good[ind_topk]

    return images_topk, boxes_topk


def get_boxes_from_detection(det_res, thr=0.95):
    r"""
    Derive the prediction boxes and assign the average box for the missed cases.
    Args:
        det_res: List[dict] and dict should have the following keys and values:
            'scores': torch.FloatTensor (N)
            'boxes': torch.FloatTensor (N,4)
        thr: float32
    Returns:
        boxes_res: torch.FloatTensor (k,4)
    """
    # select the highest score box for each case
    boxes_list, scores_list = [], []
    for res in det_res:
        scores, boxes = res['scores'], res['boxes']
        # nothing in this image
        if scores.shape[0] == 0:
            scores_list.append(-1)
            boxes_list.append(None)
        # at least one box in this image
        else:
            ind_best = torch.argmax(scores)
            score_best = scores[ind_best]
            scores_list.append(score_best)
            box_best = boxes[ind_best]
            boxes_list.append(box_best)
    # count test
    num_detection = 0
    for score in scores_list:
        if score > thr: num_detection += 1
    # refine
    if num_detection == 0:
        box_avg = torch.zeros((4), dtype=torch.float32)
        n = 0
        for box in boxes_list:
            if box is not None: 
                box_avg += box
                n += 1
        assert n != 0
        box_avg /= n
    else:
        box_avg = torch.zeros((4), dtype=torch.float32)
        n = 0
        for i, score in enumerate(scores_list):
            if score > thr:
                box_avg += boxes_list[i]
                n += 1
            else:
                boxes_list[i] = None
        assert n != 0
        box_avg /= n

    boxes_res = [box if box is not None else box_avg for box in boxes_list]
    boxes_res = torch.stack(boxes_res)

    return boxes_res


def visualize_detection_result(image_np, det_res):
    r"""
    Args:
        image_np: numpy.ndarray (B,3,H,W) range from [0,1]
        det_res: List[dict] and dict should have the following keys and values:
            'scores': torch.FloatTensor (N)
            'boxes': torch.FloatTensor (N,4)
    """
    B = image_np.shape[0]
    num_rows = num_cols = np.ceil(np.sqrt(B))

    fig = plt.figure('detection result')
    for i, slice_np in enumerate(image_np[:,0]):
        # convert to rgb
        slice_rgb = (slice_np * 255).astype(np.uint8)
        slice_rgb = cv2.cvtColor(slice_rgb, cv2.COLOR_GRAY2BGR)
        # draw box
        scores, boxes = det_res[i]['scores'], det_res[i]['boxes']
        if scores.shape[0] > 0:
            ind = torch.argmax(scores)
            score, box = scores[ind], boxes[ind]
            cv2.rectangle(slice_rgb, (box[0], box[1]), (box[2], box[3]), (0,255,0), 3)
            cv2.putText(slice_rgb, '{:.4f}'.format(score.item()), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        # show
        ax = fig.add_subplot(int(num_rows), int(2*num_cols), int(2*i+1))
        ax.imshow(slice_rgb, cmap=cm.gray)
        ax.axis('off')
        # intensity histogram
        ax = fig.add_subplot(int(num_rows), int(2*num_cols), int(2*i+2))
        intensities = np.ravel(slice_np)
        intensities = intensities[np.nonzero(intensities)]  # ignore the background
        ax.hist(intensities, bins=100)


def visualize_box_result(image_np, boxes):
    r"""
    Args:
        image_np: numpy.ndarray (B,3,H,W) range from [0,1]
        boxes: torch.FloatTensor (B,4)
    """
    B = image_np.shape[0]
    num_rows = num_cols = np.ceil(np.sqrt(B))

    fig = plt.figure('detection refined result')
    for i, slice_np in enumerate(image_np[:,0]):
        # convert to rgb
        slice_rgb = (slice_np * 255).astype(np.uint8)
        slice_rgb = cv2.cvtColor(slice_rgb, cv2.COLOR_GRAY2BGR)
        # draw box
        box = boxes[i]
        cv2.rectangle(slice_rgb, (box[0], box[1]), (box[2], box[3]), (0,255,0), 3)
        # show
        ax = fig.add_subplot(int(num_rows), int(2*num_cols), int(2*i+1))
        ax.imshow(slice_rgb, cmap=cm.gray)
        ax.axis('off')
        # intensity histogram
        ax = fig.add_subplot(int(num_rows), int(2*num_cols), int(2*i+2))
        intensities = np.ravel(slice_np)
        intensities = intensities[np.nonzero(intensities)]  # ignore the background
        ax.hist(intensities, bins=100)


def visualize_patches(patches):
    r"""
    Args:
        patches: numpy.ndarray (5,H,W) range from [-1,1]
    """
    fig = plt.figure('patches')
    for i, patch_np in enumerate(patches):
        patch_gray = (patch_np * 0.5 + 0.5) * 255
        patch_gray = patch_gray.astype(np.uint8)
        # show
        ax = fig.add_subplot(2, 5, int(i+1))
        ax.imshow(patch_gray, cmap=cm.gray)
        ax.axis('off')
        # intensity histogram
        ax = fig.add_subplot(2, 5, int(i+6))
        intensities = np.ravel(patch_np)
        ax.hist(intensities, bins=100)
