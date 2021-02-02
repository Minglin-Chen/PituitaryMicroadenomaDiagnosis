# Pituitary Microadenoma Diagnosis using Deep Learning

This repository contains the source code of PM-CAD system which is described in our paper "*Automatic diagnosis of pituitary microadenoma from magnetic resonance imaging using deep learning algorithms*". The PM-CAD system is a computer-aided diagnosis system aiming to diagnose pituitary microadenoma from MRI using deep learning techniques. 

### Dependencies

- pytorch >= 1.6.0
- SimpleITK >= 1.1.0
- albumentations >= 0.4.3
- tensorflow >= 2.0.0
- numpy >=1.16.3
- opencv-Python >= 4.1.2.30
- matplotlib >= 2.2.2
- tqdm >= 4.46.0

### Preparation

Download the checkpoints from BaiduYunPan [[Link](https://pan.baidu.com/s/17LjzK6B2-di2tIQ-6aRVKQ)] with code `frnp`.

### Demo

We provide a snippet code to demonstrate how to diagnose pituitary microadenomas using our method, run by:

```shell
python demo.py --path=Dataset/Demo/pituitary_microadenoma_case
```

### Train

To train the pituitary detection model:

```sh
cd DetectionModel
sh experiment/fasterrcnn/train_FasterRCNN.sh
```

To train the microadenomas diagnosis model (i.e., PM-Net):

```shell
cd DiagnosisModel
sh experiment/panet/run_PANet.py
```

To train other microadenomas diagnosis models:

```shell
cd DiagnosisModel
sh experiment/baseline/train_VGG16.sh
sh experiment/baseline/train_ResNeXt50.sh
sh experiment/baseline/train_ResNet50.sh
sh experiment/baseline/train_GoogLeNet.sh
sh experiment/baseline/train_DenseNet169.sh
sh experiment/cnn3d/train_CNN3D.sh
```



