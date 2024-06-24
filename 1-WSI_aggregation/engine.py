# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train and eval functions used in main.py

Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/engine.py
"""
import math
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy
from einops import rearrange
import os,glob
import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader1: Iterable,data_loader2: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=False,
                    finetune=False
                    ):
    
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    outputs = []
    targets = []
    its = 0
    for data1, data2 in metric_logger.log_every(data_loader1, data_loader2, print_freq, header):
        its+=1
        samples1 = data1[0].to(device, non_blocking=True)
        samples2 = data2[0].to(device, non_blocking=True)
        target = data1[1].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):

            output = model(samples1,samples2).squeeze()

            loss = criterion(output, torch.squeeze(target.long()))
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                raise ValueError("Loss is {}, stopping training".format(loss_value))

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            if amp:
                loss_scaler(loss, optimizer, clip_grad=max_norm,parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss.backward(create_graph=is_second_order)
                if max_norm is not None and max_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
        targets.append(target)
        if len(output.size())<2:
            output = torch.unsqueeze(output,0)
        outputs.append(output)
        # torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    num_data = len(data_loader1.dataset)
    outputs = torch.cat(outputs, dim=0)

    targets = torch.squeeze(torch.cat(targets, dim=0))
    pred = outputs.argmax(dim=1)
    predlist = (pred.eq(targets).cpu().sum()/num_data).item()
    print('Training Acc:',predlist)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # del samples1, samples2, targets, outputs
    # torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader1,data_loader2, model, device, world_size, args,distributed=False, amp=False):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []
    print_freq = 50

    its = 0
    for data1, data2 in metric_logger.log_every(data_loader1, data_loader2, print_freq, header):
        its +=1
        if args.nth_fold==0:
            slide_root = '/bigdata/projects/beidi/data/tile128to256_rand100_new'
        else:
            slide_root = '/bigdata/projects/beidi/data/tile128to256_rand100_new_kfold/'+str(args.nth_fold)
        slide_test = os.path.join(slide_root, 'test')
        # slide_test = '/bigdata/projects/beidi/dataset/urine/2306/tile'
        test_name = glob.glob(slide_test + '/*' + '/*')
        test_name = sorted(test_name)
        images1 = data1[0].to(device, non_blocking=True)
        images2 = data2[0].to(device, non_blocking=True)
        
        target = data1[1].to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(images1,images2)

        # if distributed:
        #     outputs.append(concat_all_gather(output))
        #     targets.append(concat_all_gather(target))
        # else:

        outputs.append(output)
        targets.append(target)

    num_data = len(data_loader1.dataset)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.squeeze(torch.cat(targets, dim=0))

    _, pred = outputs.max(1)
    predlist = pred.eq(targets).cpu().numpy().tolist()
    # print(pred)
    # print(targets)
    false_list_b = []
    false_list_a = []
    false_list_s = []
    false_list_c = []
    for i in range(len(predlist)):
        if predlist[i] == False:
            if test_name[i].split('/')[-2] == 'benign':
                false_list_b.append(test_name[i].split('/')[-1])
            elif test_name[i].split('/')[-2] == 'atypical':
                false_list_a.append(test_name[i].split('/')[-1])
            elif test_name[i].split('/')[-2] == 'suspicious':
                false_list_s.append(test_name[i].split('/')[-1])
            elif test_name[i].split('/')[-2] == 'cancer':
                false_list_c.append(test_name[i].split('/')[-1])
    print('false list:', 'b:', false_list_b, 'a:', false_list_a, 's:', false_list_s, 'c:', false_list_c)
    Acc_lowrisk = (48-len(false_list_b)-len(false_list_a))/48
    Acc_hignrisk = (40-len(false_list_s)-len(false_list_c))/40
    Acc_b = (13-len(false_list_b))/13
    Acc_a = (35-len(false_list_a))/35
    Acc_s = (27-len(false_list_s))/27
    Acc_c = (13-len(false_list_c))/13
    Evaluation = [Acc_lowrisk,Acc_hignrisk,Acc_b,Acc_a,Acc_s,Acc_c]
    # print("Acc low-risk: {l}, Acc high-risk: {h}".format(l=(48-len(false_list_b)-len(false_list_a))/48,h=(40-len(false_list_s)-len(false_list_c))/40))
    # print("Acc b: {b}, Acc a: {a}, Acc s: {s}, Acc c: {c}".format(b=(13-len(false_list_b))/13,a=(35-len(false_list_a))/35,s=(27-len(false_list_s))/27,c=(13-len(false_list_c))/13))
    
    from sklearn.metrics import confusion_matrix
    print('Confusion Matrix:')
    confu_matrix = confusion_matrix(targets.tolist(), pred.tolist())
    print(confu_matrix)
    AUC = roc_auc_score(targets.tolist(), pred.tolist())
    Recall = recall_score(targets.tolist(), pred.tolist())
    Precision = precision_score(targets.tolist(), pred.tolist())
    F1 = f1_score(targets.tolist(), pred.tolist())
    Sensitivity = confu_matrix[0, 0] / (confu_matrix[0, 0] + confu_matrix[1, 0])
    Specificity = confu_matrix[1, 1] / (confu_matrix[0, 1] + confu_matrix[1, 1])
    # print('AUC:', roc_auc_score(targets.tolist(), pred.tolist()))
    # print('Recall:', recall_score(targets.tolist(), pred.tolist()))
    # print('Precision:', precision_score(targets.tolist(), pred.tolist()))
    # print('F1 score:', f1_score(targets.tolist(), pred.tolist()))
    # print('Sensitivity:', confu_matrix[0, 0] / (confu_matrix[0, 0] + confu_matrix[1, 0]))
    # print('Specificity:', confu_matrix[1, 1] / (confu_matrix[0, 1] + confu_matrix[1, 1]))


    real_acc1, real_acc5 = accuracy(outputs[:num_data], targets[:num_data], topk=(1, 5))
    real_loss = criterion(outputs, targets.long())
    metric_logger.update(loss=real_loss.item())
    metric_logger.meters['acc1'].update(real_acc1.item())
    metric_logger.meters['acc5'].update(real_acc5.item())
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},AUC,Recall,Precision ,F1,Sensitivity,Specificity,Evaluation # Derive ratio of correct predictions.


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')
    return output
