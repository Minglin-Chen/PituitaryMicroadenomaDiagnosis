import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnext50_32x4d


class PANet(nn.Module):

    def __init__(self, 
                 in_channels=5, 
                 num_classes=2, 
                 p_dropout=0,
                 pretrained=False,
                 loss_name='label_smooth'):
        super(PANet, self).__init__()
        # base model
        base_model = resnet18(pretrained=pretrained)

        # stem
        out_channels = base_model.conv1.out_channels
        self.stem = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            base_model.bn1, 
            base_model.relu,
            # base_model.maxpool
        )

        # layers
        self.layers = nn.Sequential(
            base_model.layer1, 
            base_model.layer2, 
            base_model.layer3,
            # base_model.layer4
        )

        # attention layer
        final_layer = self.layers[-1][-1]
        self.final_channels = \
            final_layer.conv3.out_channels if hasattr(final_layer, 'conv3') else final_layer.conv2.out_channels
        self.layer_attn = nn.Sequential(
            nn.Conv2d(self.final_channels, 256, 1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1, bias=False), 
            nn.Sigmoid())

        # fully connection
        self.fc = nn.Sequential(
            nn.Linear(self.final_channels, 128), 
            nn.Dropout(p_dropout), 
            nn.ReLU(inplace=True), 
            nn.Linear(128, num_classes))

        # staff
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
        x = self.stem(x)
        x = self.layers(x)
        # attention
        attention = self.layer_attn(x)
        x = x * attention
        # logit
        feature = F.adaptive_avg_pool2d(x, 1).reshape(-1,self.final_channels)
        logit = self.fc(feature)
        pred_dict = {'logit': logit, 'attention': attention}
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
            loss_name: ce
        """
        # unpack
        logit = pred_dict['logit']
        label = gt_dict['label']

        loss_dict = {}
        if loss_name == 'label_smooth':
            loss_ls = self.loss_label_smooth(logit, label, 0.1)
            loss_total = loss_ls
            loss_dict['total'] = loss_total
            loss_dict['ls'] = loss_ls
        elif loss_name == 'ce':
            loss_ce = F.cross_entropy(logit, label)
            loss_total = loss_ce
            loss_dict['total'] = loss_total
            loss_dict['ce'] = loss_ce
        else:
            raise NotImplementedError
        return loss_dict

    def loss_label_smooth(self, logit, label, smoothing=0.1):
        r"""
        Args:
            logit: torch.FloatTensor (B,C)
            label: torch.LongTensor (B)
        """
        num_class = logit.shape[1]
        # smoothen weight
        weight = torch.empty_like(logit)
        weight.fill_(smoothing / (num_class-1))
        weight.scatter_(1, label.unsqueeze(1), 1.-smoothing)
        # log probability
        logprob = F.log_softmax(logit, dim=1)
        # total
        loss = torch.sum(-weight*logprob, dim=1)
        loss = loss.mean()

        return loss


if __name__=='__main__':

    # 1. fake data
    data_dict = {
        'input': torch.rand(8,5,256,256).cuda(),
        'label': torch.randint(2, (8,), dtype=torch.long).cuda()}
    # 2. model
    model = PANet(in_channels=5, num_classes=2, pretrained=True).cuda()
    model.eval()
    # 3. forward
    ret_dict = model(data_dict)
    # 4. info
    print(ret_dict)