import argparse
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from collections import OrderedDict
import torch

from DetectionModel.model import model_provider as detection_model_provider
from DiagnosisModel.model import model_provider as diagnosis_model_provider

from utils import load_dicom_series_4D, preprocessing_on_dicom_4D, topk_boxes_from_detection
from utils import load_jpg_slices, preprocessing_on_slice, get_boxes_from_detection
from utils import visualize_detection_result, visualize_box_result, visualize_patches


@torch.no_grad()
def predict_on_dicom(directory, detection_model, diagnosis_model, visualization=False):

    # 1. safe load dicom data
    image_np = load_dicom_series_4D(directory, series_description='T1W_TSE_Dyn')
    assert isinstance(image_np, np.ndarray), directory
    # 2. preprocessing
    image_np_list = preprocessing_on_dicom_4D(image_np)
    # 3. detection
    # forward (saving GPU memory)
    proba_list = []
    for image_np in image_np_list:
        ntime = image_np.shape[0]
        image_np = image_np[ntime//2-2:ntime//2+3]

        image_tensor = torch.from_numpy(image_np)
        det_res = [detection_model([slice_tensor.cuda()])[0] for slice_tensor in image_tensor]
        det_res = [{k:v.cpu() for k,v in res.items()} for res in det_res]
        if visualization: visualize_detection_result(image_np, det_res)
        
        try:
            boxes = get_boxes_from_detection(det_res, thr=0.5)
            if visualization: visualize_box_result(image_np, boxes)
            # 4. form the patches
            patches = []
            for image, box in zip(image_np, boxes):
                image_crop = image[0,int(box[1]):int(box[3])+1,int(box[0]):int(box[2])+1]
                image_resize = cv2.resize(image_crop, (256,256))
                image_normalize = image_resize * 2. - 1.
                patches.append(image_normalize)
            patches = np.stack(patches)
            if visualization: visualize_patches(patches)
            # 5. diagnosis
            patches_tensor = torch.from_numpy(patches)
            ret_res = diagnosis_model({'input': patches_tensor.cuda()[None]})
            logit = ret_res['logit']
            proba = logit.softmax(dim=1)[0,1].item()
            proba_list.append(proba)
        except:
            proba_list.append(0.0)

        if visualization: plt.show()

    return np.max(proba_list)


@torch.no_grad()
def predict_on_slices(directory, detection_model, diagnosis_model, visualization=False):

    # 1. load slices
    slices_np = load_jpg_slices(directory)
    # 2. preprocessing
    image_np = preprocessing_on_slice(slices_np)
    # 3. detection
    # forward (saving GPU memory)
    image_tensor = torch.from_numpy(image_np)
    det_res = [detection_model([slice_tensor.cuda()])[0] for slice_tensor in image_tensor]
    det_res = [{k:v.cpu() for k,v in res.items()} for res in det_res]
    if visualization: visualize_detection_result(image_np, det_res)
    # get the boxes
    boxes = get_boxes_from_detection(det_res, thr=0.5)
    if visualization: visualize_box_result(image_np, boxes)
    # 4. form the patches
    patches = []
    for image, box in zip(image_np, boxes):
        image_crop = image[0,int(box[1]):int(box[3])+1,int(box[0]):int(box[2])+1]
        image_resize = cv2.resize(image_crop, (256,256))
        image_normalize = image_resize * 2. - 1.
        patches.append(image_normalize)
    patches = np.stack(patches)
    if visualization: visualize_patches(patches)
    # 5. diagnosis
    patches_tensor = torch.from_numpy(patches)
    ret_res = diagnosis_model({'input': patches_tensor.cuda()[None]})
    logit = ret_res['logit']
    proba = logit.softmax(dim=1)[0,1].item()

    if visualization: plt.show()

    return proba


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='slice', required=False)
    parser.add_argument('--path', type=str, default='Dataset/Demo/pituitary_microadenoma_case', required=False)
    parser.add_argument('--visualize', type=bool, default=False, required=False)
    args = parser.parse_args()

    # load well-trained models
    detection_model = detection_model_provider('FasterRCNN', num_classes=2).cuda()
    state_dict = torch.load('Checkpoint/FasterRCNN.pt')
    detection_model.load_state_dict(state_dict, strict=True)
    detection_model.eval()

    diagnosis_model = diagnosis_model_provider('PANet').cuda()
    state_dict = torch.load('Checkpoint/PANet.pt')
    state_dict = OrderedDict({k[7:]:v for k,v in state_dict.items()})
    diagnosis_model.load_state_dict(state_dict, strict=True)
    diagnosis_model.eval()
    threshold = 0.52828431129455566406 # optimal threshold selected from ROC

    if args.mode == 'slice':
        proba = predict_on_slices(args.path, detection_model, diagnosis_model, args.visualize)
    elif args.mode == 'dicom':
        proba = predict_on_dicom(args.path, detection_model, diagnosis_model, args.visualize)
    else:
        raise NotImplementedError

    if proba>=threshold:
        print('Diagnosis result: pituitary microadenoma')
    else:
        print('Diagnosis result: normal')