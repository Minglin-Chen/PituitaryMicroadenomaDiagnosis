import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.vgg import vgg16, vgg16_bn
from torchvision.models.resnet import resnet18, resnet50, resnext50_32x4d
from torchvision.models.densenet import densenet169
from torchvision.models.googlenet import googlenet


class CNN(nn.Module):
    r"""
    Wrapper for existing model
    Args:
        model_name: str
        in_channels: int32
        num_classes: int32
        pretrained: bool, whether to load the pretrained weights
        loss_name: str
    """
    def __init__(self, 
                 model_name='resnet18', 
                 in_channels=5, 
                 num_classes=2, 
                 pretrained=False, 
                 loss_name='ce'):
        super(CNN, self).__init__()

        # load the existing model
        self.model = eval(model_name)(pretrained=pretrained)
        # change the model
        if 'vgg' in model_name:
            out_channels = self.model.features[0].out_channels
            self.model.features[0] = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif 'resnet' in model_name:
            out_channels = self.model.conv1.out_channels
            self.model.conv1 = nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif 'resnext' in model_name:
            out_channels = self.model.conv1.out_channels
            self.model.conv1 = nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif 'densenet' in model_name:
            out_channels = self.model.features[0].out_channels
            self.model.features[0] = nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif 'googlenet' in model_name:
            self.model.aux_logits = False
            self.model._transform_input = lambda x:x
            out_channels = self.model.conv1.conv.out_channels
            self.model.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_channels, eps=0.001),
                nn.ReLU(inplace=True))
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            
        # staff
        self.model_name = model_name
        self.loss_name = loss_name

    def forward(self, data_dict):
        r"""
        Args:
            data_dict: dict should have the following keys and values:
                'input': torch.FloatTensor (B,C,W,H)
        """
        # unpack
        x = data_dict['input']
        # forward
        logit = self.model(x)
        pred_dict = {'logit': logit}
        # train or inference
        if self.training:
            gt_dict = {'label': data_dict['label']}
            loss_dict = self.loss(pred_dict, gt_dict, self.loss_name)
            ret_dict = loss_dict
        else:
            ret_dict = pred_dict
            if 'label' in data_dict.keys():
                gt_dict = {'label': data_dict['label']}
                loss_dict = self.loss(pred_dict, gt_dict, self.loss_name)
                ret_dict = {**ret_dict, **loss_dict}
        
        return ret_dict

    def loss(self, pred_dict, gt_dict, loss_name):
        r"""
        Args:
            pred_dict: dict
            gt_dict: dict
            loss_name: str
        """
        # unpack
        logit = pred_dict['logit']
        label = gt_dict['label']

        loss_dict = {}
        if loss_name == 'ce':
            loss_ce = F.cross_entropy(logit, label)
            loss_total = loss_ce
            loss_dict['total'] = loss_total
            loss_dict['ce'] = loss_ce
        else:
            raise NotImplementedError
        return loss_dict


if __name__=='__main__':

    # 1. fake data
    data_dict = {
        'input': torch.rand(8,5,256,256).cuda(),
        'label': torch.randint(2, (8,), dtype=torch.long).cuda()}
    # 2. model
    model = CNN(model_name='resnet18', in_channels=5, num_classes=2, pretrained=False).cuda()
    model.eval()
    # 3. forward
    ret_dict = model(data_dict)
    # 4. info
    print(ret_dict)