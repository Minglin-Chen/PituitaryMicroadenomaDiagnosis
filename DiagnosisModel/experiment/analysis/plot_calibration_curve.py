import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.calibration import calibration_curve
from scipy.stats import linregress

Pastel1_CM = plt.get_cmap('Pastel1')
fmts = ['.--', '^--', '*--', '>--', 'x--','|--', 'd-']
colors = [Pastel1_CM(0), Pastel1_CM(1), Pastel1_CM(2), Pastel1_CM(3), Pastel1_CM(4), Pastel1_CM(6), 'orange']


def plot_multi_calibration_curves(labels, probas, names, n_bins=5):

    # setting
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['savefig.jpeg_quality'] = 100

    plt.figure(figsize=(4,3), dpi=300)
    plt.plot([0,1], [0,1], 'k:', label='Perfectly calibrated')

    for i, (label, proba, name) in enumerate(zip(labels, probas, names)):
        freq, pred = calibration_curve(label, proba, normalize=False, n_bins=n_bins, strategy='quantile')
        slope, intercept, _, _, _ = linregress(pred, freq)
        plt.plot(pred, freq, fmts[i], color=colors[i], label=name+' (s:{:.2f} i:{:.3f})'.format(slope, intercept))

    plt.xlabel('Prediceted probability')
    plt.xlim([-0.02,1.02])
    plt.ylabel('Observed frequency')
    plt.ylim([-0.02,1.02])
    plt.legend(loc="lower right")
    plt.tight_layout()
    # plt.savefig('calibration_py.svg')
    plt.show()


def plot_calibration_curve(label, proba, n_bins=5):

    # setting
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 6})
    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['savefig.jpeg_quality'] = 100

    plt.figure(figsize=(3,2.25), dpi=300)
    plt.plot([0,1], [0,1], 'k:', label='Perfectly calibrated')

    freq, pred = calibration_curve(label, proba, normalize=False, n_bins=n_bins, strategy='quantile')
    plt.plot(pred, freq, '+-', label='model')

    plt.xlabel('Prediceted probability')
    plt.xlim([-0.02,1.02])
    plt.ylabel('Observed frequency')
    plt.ylim([-0.02,1.02])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def align_probability(label_train, proba_train, proba_val, n_bins=5):

    # sort
    ind = np.argsort(proba_train)
    proba_train, label_train = proba_train[ind], label_train[ind]

    # stone
    proba_stones = [ proba_train[int(len(proba_train)/n_bins*(i+1)) - 1] for i in range(n_bins) ]

    # ratio
    freq, pred = calibration_curve(label_train, proba_train, normalize=False, n_bins=n_bins, strategy='quantile')
    proba_ratio = []
    proba_trans = [0]
    for freq_value, pred_value, proba_value in zip(freq, pred, proba_stones):
        trans = proba_trans[-1]
        ratio = freq_value / (pred_value - trans)
        proba_ratio.append(ratio)
        trans_next = proba_value - (proba_value-trans) * ratio
        proba_trans.append(trans_next)
    
    # algin
    low = 0
    for high, ratio, trans in zip(proba_stones, proba_ratio, proba_trans[:-1]):
        proba_val[(proba_val>=low) & (proba_val<high)] -= trans 
        proba_val[(proba_val>=low) & (proba_val<high)] *= ratio
        low = high

    proba_val[proba_val>1] = 1
    proba_val[proba_val<0] = 0

    return proba_val


def load_and_align(json_path):

    with open(json_path[:-5]+'_train.json', 'r') as f:
        sample_list = json.load(f)
    label_train, proba_train = np.asarray(sample_list['label']), np.asarray(sample_list['proba'])
    with open(json_path, 'r') as f:
        sample_list = json.load(f)
    label_val, proba_val = np.asarray(sample_list['label']), np.asarray(sample_list['proba'])

    proba_val = align_probability(label_train, proba_train, proba_val, n_bins=15)

    return label_val.tolist(), proba_val.tolist()


if __name__=='__main__':

    labels, probas, names = [], [], []

    label, proba = load_and_align('DiagnosisModel/result/baseline/densenet169_inference.json')
    labels.append(label)
    probas.append(proba)
    names.append('DenseNet')

    label, proba = load_and_align('DiagnosisModel/result/baseline/googlenet_inference.json')
    labels.append(label)
    probas.append(proba)
    names.append('GoogLeNet')

    label, proba = load_and_align('DiagnosisModel/result/CNN3D/CNN3D_inference.json')
    labels.append(label)
    probas.append(proba)
    names.append('3D-CNN')

    label, proba = load_and_align('DiagnosisModel/result/baseline/vgg16_bn_inference.json')
    labels.append(label)
    probas.append(proba)
    names.append('VGG')

    label, proba = load_and_align('DiagnosisModel/result/baseline/resnet50_inference.json')
    labels.append(label)
    probas.append(proba)
    names.append('ResNet')

    label, proba = load_and_align('DiagnosisModel/result/baseline/resnext50_32x4d_inference.json')
    labels.append(label)
    probas.append(proba)
    names.append('ResNeXt')

    label, proba = load_and_align('DiagnosisModel/result/PMNet/PANet_inference.json')
    labels.append(label)
    probas.append(proba)
    names.append('PM-Net (ours)')

    plot_multi_calibration_curves(labels, probas, names, n_bins=5)