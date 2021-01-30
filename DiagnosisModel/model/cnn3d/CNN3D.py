import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNet3D import UNet3D


class CNN3D(nn.Module):
    r"""
    3D CNN
    Args:
        num_classes: int32
        pretrained: str
        loss_name: str
    """
    def __init__(self, 
                 num_classes=2, 
                 pretrained='',
                 loss_name='ce'):
        super(CNN3D, self).__init__()
        # pretrained
        backbone = UNet3D()
        if pretrained:
            print('Load weight from {}'.format(pretrained))
            state_dict = torch.load(pretrained)['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            backbone.load_state_dict(unParalled_state_dict)
        # backbone
        self.down_tr64 = backbone.down_tr64
        self.down_tr128 = backbone.down_tr128
        self.down_tr256 = backbone.down_tr256
        self.down_tr512 = backbone.down_tr512
        # fc
        self.fc = nn.Sequential(
            nn.Linear(512, 1024, bias=True), nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes, bias=True))

        # staff
        self.loss_name = loss_name

    def forward(self, data_dict):
        r"""
        Args:
            data_dict: dict should have the following keys and values:
                'input': torch.FloatTensor (B,1,C,W,H)
        """
        # unpack
        x = data_dict['input']
        # backbone
        x, _ = self.down_tr64(x)
        x, _ = self.down_tr128(x)
        x, _ = self.down_tr256(x)
        x, _ = self.down_tr512(x)
        # feature (B,512)
        feature = F.adaptive_avg_pool3d(x, 1).reshape(-1,512)
        # logit (B,C)
        logit = self.fc(feature)
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
        'input': torch.rand(16,1,5,256,256).cuda(),
        'label': torch.randint(2, (16,), dtype=torch.long).cuda()}
    # 2. model
    model = CNN3D(num_classes=2, pretrained='').cuda()
    # 3. forward
    ret_dict = model(data_dict)
    # 4. info
    print(ret_dict)

