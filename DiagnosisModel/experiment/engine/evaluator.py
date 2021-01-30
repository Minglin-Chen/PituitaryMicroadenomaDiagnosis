import time
import torch
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

class evaluator():

    def __init__(self, model, dataloader, logger=None):
        self.model = model
        self.dataloader = dataloader
        self.logger = logger

    @torch.no_grad()
    def __call__(self, epoch=0):
        # setup
        self.model.eval()

        # loop
        loss = 0.0
        confusion = torch.zeros((2, 2)).double().cuda()
        proba_list, label_list = [], []
        for data_dict in tqdm(self.dataloader):
                
            # 1. forward
            ret_dict = self.model(data_dict)

            # 2. loss
            loss += ret_dict['total'].mean().item()
            
            # 3. prediction
            pred = torch.argmax(ret_dict['logit'], dim=1)

            # 4. statistic
            pred_flat, label_flat = pred.view(-1), data_dict['label'].view(-1).cuda()

            accumulate = pred_flat + label_flat
            tp = accumulate == 2
            tn = accumulate == 0
            fp = pred_flat > label_flat
            fn = pred_flat < label_flat

            # 5. update
            confusion = self._update_confusion(confusion, tp, tn, fp, fn)
            proba_list += ret_dict['logit'].softmax(dim=1)[:,1].tolist()
            label_list += (data_dict['label']).tolist()
            
        eval_string = 'current metric: '
        metric_dict = {}

        # Loss
        loss /= len(self.dataloader)
        eval_string += '[loss {:.4f}] '.format(loss)
        if self.logger is not None: self.logger.add_scalar('loss', loss, epoch)
        metric_dict['loss'] = loss

        # AUC
        auc = roc_auc_score(label_list, proba_list)
        eval_string += '[auc {:.4f}] '.format(auc)
        if self.logger is not None: self.logger.add_scalar('auc', auc, epoch)
        metric_dict['auc'] = auc
        metric_dict['main'] = auc

        tn, fp, fn, tp = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
        # F1score
        f1score = 1.0 if (2 * tp + fn + fp) == 0 else 2 * tp / (2 * tp + fn + fp)
        eval_string += '[f1score {:.4f}] '.format(f1score)
        if self.logger is not None: self.logger.add_scalar('f1score', f1score, epoch)
        metric_dict['f1score'] = f1score

        # Accuracy
        acc = (tn + tp) / (tn + fp + fn + tp)
        eval_string += '[acc {:.4f}] '.format(acc)
        if self.logger is not None: self.logger.add_scalar('acc', acc, epoch)
        metric_dict['acc'] = acc
        
        # Sensitivity
        sensitivity = 1.0 if tp + fn == 0 else tp / (tp + fn)
        eval_string += '[sensitivity {:.4f}] '.format(sensitivity)
        if self.logger is not None: self.logger.add_scalar('sensitivity', sensitivity, epoch)
        metric_dict['sensitivity'] = sensitivity

        # Specificity
        specificity = 1.0 if fp + tn == 0 else tn / (fp + tn)
        eval_string += '[specificity {:.4f}] '.format(specificity)
        if self.logger is not None: self.logger.add_scalar('specificity', specificity, epoch)
        metric_dict['specificity'] = specificity

        # PPV
        ppv = 1.0 if tp + fp == 0 else tp / (tp + fp)
        eval_string += '[ppv {:.4f}] '.format(ppv)
        if self.logger is not None: self.logger.add_scalar('ppv', ppv, epoch)
        metric_dict['ppv'] = ppv

        # npv
        npv = 1.0 if tn + fn == 0 else tn / (tn + fn)
        eval_string += '[npv {:.4f}] '.format(npv)
        if self.logger is not None: self.logger.add_scalar('npv', npv, epoch)
        metric_dict['npv'] = npv

        # Print
        print(eval_string)

        return metric_dict

    def _update_confusion(self, matrix, tp, tn, fp, fn):

        matrix[1, 1] += tp.sum()
        matrix[0, 0] += tn.sum()
        matrix[0, 1] += fp.sum()
        matrix[1, 0] += fn.sum()

        return matrix