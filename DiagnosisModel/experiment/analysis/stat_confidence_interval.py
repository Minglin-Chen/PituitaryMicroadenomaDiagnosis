import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc

########################################################################
# helper function used for Clopper-Pearson confidence intervals (Start)
# refer to `https://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals`
########################################################################
def binP(N, p, x1, x2):
    p = float(p)
    q = p/(1-p)
    k = 0.0
    v = 1.0
    s = 0.0
    tot = 0.0

    while(k<=N):
        tot += v
        if(k >= x1 and k <= x2):
            s += v
        if(tot > 10**30):
            s = s/10**30
            tot = tot/10**30
            v = v/10**30
        k += 1
        v = v*q*(N+1-k)/k
    return s/tot


def calcBin(vx, vN, vCL = 95):
    '''
    Calculate the exact confidence interval for a binomial proportion

    Usage:
    >>> calcBin(13,100)    
    (0.07107391357421874, 0.21204372406005856)
    >>> calcBin(4,7)   
    (0.18405151367187494, 0.9010086059570312)
    ''' 
    vx = float(vx)
    vN = float(vN)
    #Set the confidence bounds
    vTU = (100 - float(vCL))/2
    vTL = vTU

    vP = vx/vN
    if(vx==0):
        dl = 0.0
    else:
        v = vP/2
        vsL = 0
        vsH = vP
        p = vTL/100

        while((vsH-vsL) > 10**-5):
            if(binP(vN, v, vx, vN) > p):
                vsH = v
                v = (vsL+v)/2
            else:
                vsL = v
                v = (v+vsH)/2
        dl = v

    if(vx==vN):
        ul = 1.0
    else:
        v = (1+vP)/2
        vsL =vP
        vsH = 1
        p = vTU/100
        while((vsH-vsL) > 10**-5):
            if(binP(vN, v, 0, vx) < p):
                vsH = v
                v = (vsL+v)/2
            else:
                vsL = v
                v = (v+vsH)/2
        ul = v
    return (dl, ul)
########################################################################
# helper function used for Clopper-Pearson confidence intervals (End)
########################################################################


def stat_clopper_pearson_confidence_intervals(label, proba, thresh=0.5):
    r"""
    Args:
        label: List[int32]
        proba: List[float32]
        thresh: float32
        num_sample: int32
    """
    # convert to numpy.ndarray
    label = np.array(label, dtype=np.bool)
    proba = np.array(proba)

    ####################################################
    # performance
    ####################################################    
    pred = proba >= thresh
    # tp tn fp fn
    tp = np.bitwise_and(pred, label).sum()
    tn = np.bitwise_and(~pred, ~label).sum()
    fp = np.bitwise_and(pred, ~label).sum()
    fn = np.bitwise_and(~pred, label).sum()
    print('tp tn fp fn: {} {} {} {}'.format(tp, tn, fp, fn))
    # F1-score
    f1score = 1.0 if (2 * tp + fn + fp) == 0 else 2 * tp / (2 * tp + fn + fp)
    low_ci, high_ci = calcBin(2 * tp, 2 * tp + fn + fp)
    print('F1-score: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(f1score, 2*tp, 2*tp+fn+fp, low_ci, high_ci))
    # Accuracy
    acc = (tn + tp) / (tn + fp + fn + tp)
    low_ci, high_ci = calcBin(tn + tp, tn + fp + fn + tp)
    print('Accuracy: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(acc, tn+tp, tn+fp+fn+tp, low_ci, high_ci))
    # Sensitivity
    sensitivity = 1.0 if tp + fn == 0 else tp / (tp + fn)
    low_ci, high_ci = calcBin(tp, tp + fn)
    print('Sensitivity: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(sensitivity, tp, tp+fn, low_ci, high_ci))
    # PPV
    ppv = 1.0 if tp + fp == 0 else tp / (tp + fp)
    low_ci, high_ci = calcBin(tp, tp + fp)
    print('PPV: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}] '.format(ppv, tp, tp+fp, low_ci, high_ci))
    # Specificity
    specificity = 1.0 if fp + tn == 0 else tn / (fp + tn)
    low_ci, high_ci = calcBin(tn, fp + tn)
    print('Specificity: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(specificity, tn, fp+tn, low_ci, high_ci))
    # NPV
    npv = 1.0 if tn + fn == 0 else tn / (tn + fn)
    low_ci, high_ci = calcBin(tn, tn + fn)
    print('NPV: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(npv, tn, tn+fn, low_ci, high_ci))


def stat_clopper_pearson_confidence_intervals2(tp, fp, fn, tn):
    r"""
    Args:
        tp: int32
        fp: int32
        fn: int32
        tn: int32
    """
    ####################################################
    # performance
    ####################################################
    print('tp tn fp fn: {} {} {} {}'.format(tp, tn, fp, fn))
    # F1-score
    f1score = 1.0 if (2 * tp + fn + fp) == 0 else 2 * tp / (2 * tp + fn + fp)
    low_ci, high_ci = calcBin(2 * tp, 2 * tp + fn + fp)
    print('F1-score: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(f1score, 2*tp, 2*tp+fn+fp, low_ci, high_ci))
    # Accuracy
    acc = (tn + tp) / (tn + fp + fn + tp)
    low_ci, high_ci = calcBin(tn + tp, tn + fp + fn + tp)
    print('Accuracy: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(acc, tn+tp, tn+fp+fn+tp, low_ci, high_ci))
    # Sensitivity
    sensitivity = 1.0 if tp + fn == 0 else tp / (tp + fn)
    low_ci, high_ci = calcBin(tp, tp + fn)
    print('Sensitivity: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(sensitivity, tp, tp+fn, low_ci, high_ci))
    # PPV
    ppv = 1.0 if tp + fp == 0 else tp / (tp + fp)
    low_ci, high_ci = calcBin(tp, tp + fp)
    print('PPV: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}] '.format(ppv, tp, tp+fp, low_ci, high_ci))
    # Specificity
    specificity = 1.0 if fp + tn == 0 else tn / (fp + tn)
    low_ci, high_ci = calcBin(tn, fp + tn)
    print('Specificity: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(specificity, tn, fp+tn, low_ci, high_ci))
    # NPV
    npv = 1.0 if tn + fn == 0 else tn / (tn + fn)
    low_ci, high_ci = calcBin(tn, tn + fn)
    print('NPV: \n{:.4f} ({}/{})\n[{:.4f}~{:.4f}]'.format(npv, tn, tn+fn, low_ci, high_ci))


def stat_bootstrap_confidence_intervals(label, proba, thresh=0.5, num_resample=50000):
    r"""
    Args:
        label: List[int32]
        proba: List[float32]
        thresh: float32
        num_sample: int32
    """
    # convert to numpy.ndarray
    label = np.array(label, dtype=np.bool)
    proba = np.array(proba)

    ####################################################
    # performance
    ####################################################    
    pred = proba >= thresh
    # tp tn fp fn
    tp = np.bitwise_and(pred, label).sum()
    tn = np.bitwise_and(~pred, ~label).sum()
    fp = np.bitwise_and(pred, ~label).sum()
    fn = np.bitwise_and(~pred, label).sum()
    # AUC
    fpr, tpr, _ = roc_curve(label, proba)
    auc_value = auc(fpr, tpr)
    print('AUC: {:.4f}'.format(auc_value))
    # # F1-score
    # f1score = 1.0 if (2 * tp + fn + fp) == 0 else 2 * tp / (2 * tp + fn + fp)
    # print('F1-score: {:.4f}'.format(f1score))
    # # Accuracy
    # acc = (tn + tp) / (tn + fp + fn + tp)
    # print('Accuracy: {:.4f}'.format(acc))
    # # Sensitivity
    # sensitivity = 1.0 if tp + fn == 0 else tp / (tp + fn)
    # print('Sensitivity: {:.4f}'.format(sensitivity))
    # # Specificity
    # specificity = 1.0 if fp + tn == 0 else tn / (fp + tn)
    # print('Specificity: {:.4f}'.format(specificity))
    # # PPV
    # ppv = 1.0 if tp + fp == 0 else tp / (tp + fp)
    # print('PPV: {:.4f}'.format(ppv))
    # # NPV
    # npv = 1.0 if tn + fn == 0 else tn / (tn + fn)
    # print('NPV: {:.4f}'.format(npv))

    ####################################################
    # performance 95% CI
    ####################################################
    # resample
    num_data = label.shape[0]
    resample_index = np.random.choice(num_data, size=(num_resample, num_data), replace=True)
    label_resample, proba_resample = label[resample_index], proba[resample_index]

    perf_dict = {}
    for label, proba in zip(label_resample, proba_resample):
        pred = proba >= thresh
        # tp tn fp fn
        tp = np.bitwise_and(pred, label).sum()
        tn = np.bitwise_and(~pred, ~label).sum()
        fp = np.bitwise_and(pred, ~label).sum()
        fn = np.bitwise_and(~pred, label).sum()
        # AUC
        fpr, tpr, _ = roc_curve(label, proba)
        auc_value = auc(fpr, tpr)
        if 'AUC' not in perf_dict.keys():
            perf_dict['AUC'] = []
        perf_dict['AUC'].append(auc_value)
        # # F1-score
        # f1score = 1.0 if (2 * tp + fn + fp) == 0 else 2 * tp / (2 * tp + fn + fp)
        # if 'F1-score' not in perf_dict.keys():
        #     perf_dict['F1-score'] = []
        # perf_dict['F1-score'].append(f1score)
        # # Accuracy
        # acc = (tn + tp) / (tn + fp + fn + tp)
        # if 'Accuracy' not in perf_dict.keys():
        #     perf_dict['Accuracy'] = []
        # perf_dict['Accuracy'].append(acc)
        # # Sensitivity
        # sensitivity = 1.0 if tp + fn == 0 else tp / (tp + fn)
        # if 'Sensitivity' not in perf_dict.keys():
        #     perf_dict['Sensitivity'] = []
        # perf_dict['Sensitivity'].append(sensitivity)
        # # Specificity
        # specificity = 1.0 if fp + tn == 0 else tn / (fp + tn)
        # if 'Specificity' not in perf_dict.keys():
        #     perf_dict['Specificity'] = []
        # perf_dict['Specificity'].append(specificity)
        # # PPV
        # ppv = 1.0 if tp + fp == 0 else tp / (tp + fp)
        # if 'PPV' not in perf_dict.keys():
        #     perf_dict['PPV'] = []
        # perf_dict['PPV'].append(ppv)
        # # NPV
        # npv = 1.0 if tn + fn == 0 else tn / (tn + fn)
        # if 'NPV' not in perf_dict.keys():
        #     perf_dict['NPV'] = []
        # perf_dict['NPV'].append(npv)

    # print
    for perf_name, perf_list in perf_dict.items():
        perf_list = sorted(perf_list)
        n = int(num_resample * 0.025)
        low_CI, high_CI = perf_list[n], perf_list[-n]
        print('{}: [{:.4f}~{:.4f}]'.format(perf_name, low_CI, high_CI))


def get_optimal_threshold(label, proba):
    r"""
    Get the optimal probability threshold based on Youden Index
    Args:
        label: List
        proba: List
    Returns:
        optimal_thresh: float32
        optimal_tpr: float32
        optimal_fpr: float32
    """
    fpr, tpr, thresholds = roc_curve(label, proba)
    value = tpr - fpr
    index = np.argmax(value)
    optimal_thresh = thresholds[index]
    optimal_tpr = tpr[index]
    optimal_fpr = fpr[index]
    return optimal_thresh, optimal_tpr, optimal_fpr


if __name__=='__main__':

    # fix random seed
    np.random.seed(666)

    # load data
    # with open('DiagnosisModel/result/baseline/googlenet_inference.json', 'r') as f:
    # with open('DiagnosisModel/result/baseline/vgg16_bn_inference.json', 'r') as f:
    # with open('DiagnosisModel/result/baseline/resnet50_inference.json', 'r') as f:
    # with open('DiagnosisModel/result/baseline/densenet169_inference.json', 'r') as f:
    # with open('DiagnosisModel/result/baseline/resnext50_32x4d_inference.json', 'r') as f:
    # with open('DiagnosisModel/result/CNN3D/CNN3D_inference.json', 'r') as f:
    with open('DiagnosisModel/result/PMNet/PANet_inference.json', 'r') as f:
    # with open('comparison_result.json', 'r') as f:
        sample = json.load(f)
    label, proba = sample['label'], sample['proba']

    # optimal threshold
    optimal_thresh, optimal_tpr, optimal_fpr = get_optimal_threshold(label, proba)
    print('optimal threshold: {:.20f} (tpr {:.4f} fpr {:.4f})'.format(optimal_thresh, optimal_tpr, optimal_fpr))
    # 95% CI
    # optimal_thresh = 0.52828431129455566406
    stat_bootstrap_confidence_intervals(label, proba, optimal_thresh)
    stat_clopper_pearson_confidence_intervals(label, proba, optimal_thresh)

    # tp, fp, fn, tn = 37, 11, 13, 39
    # tp, fp, fn, tn = 39, 10, 11, 40
    # tp, fp, fn, tn = 42, 6, 8, 44
    # tp, fp, fn, tn = 45, 7, 5, 43
    # tp, fp, fn, tn = 45, 4, 5, 46
    # tp, fp, fn, tn = 47, 2, 3, 48
    # stat_clopper_pearson_confidence_intervals2(tp, fp, fn, tn)