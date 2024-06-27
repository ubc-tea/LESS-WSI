import math
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import glob

from utils.checkpoint import *

from utils.func import *

# from slide_test import Graph_classification,Threshold, MLP_main,Graph_classification_multiscale
from utils.feature import get_feature_urine
from dataset.dataset_urine import UrineSlideDataset
# from slide_test_FANC import Graph_classification,Threshold, MLP_main,Graph_classification_multiscale
# from dataset.dataset_FANC import UrineSlideDataset
# from GCN_model import GCN

def run_vpu(config, loaders, NetworkPhi,nth_fold):
    """
    run VPU.
    :param config: arguments.
    :param loaders: loaders.
    :param NetworkPhi: class of the model.
    """

    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1 # highest test accuracy on test set

    # set up loaders
    (p_loader, x_loader, val_p_loader, val_x_loader, test_loader) = loaders

    # set up vpu model and dataset
    if config.dataset in ['cifar10', 'fashionMNIST', 'stl10','urine','FANC']:
        model_phi = NetworkPhi()
        # model_phi = nn.DataParallel(model_phi)
        save_dir = os.path.join('.','saved_checkpoint', '_'.join((config.dataset,'lr='+str(config.lr), 'lambda='+str(config.lam), 'alpha='+str(config.alpha), 'scale='+str(config.scale))), 'fold='+str(config.nth_fold))
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        checkpoint_path = save_dir
        mean_v_samp = torch.Tensor([])
        for p in model_phi.parameters():
            mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
        print(mean_v_samp)
    elif config.dataset in ['pageblocks', 'grid', 'avila']:
        input_size = len(p_loader.dataset[0][0])
        model_phi = NetworkPhi(input_size=input_size)
    if torch.cuda.is_available():
        model_phi = model_phi.cuda()

    # set up the optimizer
    lr_phi = config.lr
    opt_phi = torch.optim.Adam(model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99))

  
    for epoch in range(config.epochs):

        # adjust the optimizer
        if epoch <= 5 and epoch % 2 == 1:
            lr_phi /= 2
            print('Learning rate changes to',lr_phi)
            # opt_phi = torch.optim.SGD(model_phi.parameters(), lr=lr_phi, momentum=0.9)
            opt_phi = torch.optim.Adam(model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99))

        # train the model \Phi
        phi_loss, var_loss, reg_loss, phi_p_mean, phi_x_mean = train(config, model_phi, opt_phi, p_loader, x_loader)

        # evaluate the model \Phi
        val_var, test_acc, log_max_phi= evaluate(config,model_phi, x_loader, test_loader, val_p_loader, val_x_loader, epoch,
                                              phi_loss, var_loss, reg_loss)

        # assessing performance of the current model and decide whether to save it
        is_val_var_lowest = val_var < lowest_val_var
        is_test_acc_highest = test_acc > highest_test_acc
        lowest_val_var = min(lowest_val_var, val_var)
        highest_test_acc = max(highest_test_acc, test_acc)
        if is_val_var_lowest:
            # test_auc_of_best_val = test_auc
            test_acc_of_best_val = test_acc
            epoch_of_best_val = epoch
            best_model = model_phi.state_dict()
        torch.save(best_model, checkpoint_path + '/' + str(epoch) + '.pth')

    if config.get_feature:
        if config.dataset == 'urine':
            get_feature_urine(config,model_phi,nth_fold,config.VPUep)


    # if config.get_label:
        # get_vpu_label(config, log_max_phi) 
    
    # inform users model in which epoch is finally picked
    # Threshold(model_phi, config,10)
    # MLP_main(config)
  
    # Graph_classification(config,model_phi, GNN_model,nth_fold,config.VPUep)
    # GNN_model_save,GNN_best_acc = Graph_classification_multiscale(config,model_phi, GNN_model,  nth_fold)
    # print('Early stopping at {:}th epoch, test acc: {:.4f}'.format(epoch_of_best_val, test_acc_of_best_val))
    # print('Load model of epoch',epoch_of_best_val )
    # model_phi.load_state_dict(best_model)

def train(config, model_phi, opt_phi, p_loader, x_loader):
    """
    One epoch of the training of VPU.

    :param config: arguments.
    :param model_phi: current model \Phi.
    :param opt_phi: optimizer of \Phi.
    :param p_loader: loader for the labeled positive training data.
    :param x_loader: loader for training data (including positive and unlabeled)
    """
    # setup some utilities for analyzing performance
    phi_p_avg = AverageMeter()
    phi_x_avg = AverageMeter()
    phi_loss_avg = AverageMeter()
    var_loss_avg = AverageMeter()
    reg_avg = AverageMeter()

    # set the model to train mode
    model_phi.train()

    for batch_idx in range(config.val_iterations):
        try:
            data_x, _ = next(x_iter)
        except:
            x_iter = iter(x_loader)
            data_x, _ = next(x_iter)

        try:
            data_p, _ = next(p_iter)
        except:
            p_iter = iter(p_loader)
            data_p, _ = next(p_iter)

        if torch.cuda.is_available():
            data_p, data_x = data_p.cuda(), data_x.cuda()

        # calculate the variational loss
        data_all = torch.cat((data_p, data_x))
        output_phi_all,_ = model_phi(data_all)
        log_phi_all = output_phi_all[:, 1]
        idx_p = slice(0, len(data_p))
        idx_x = slice(len(data_p), len(data_all))
        log_phi_x = log_phi_all[idx_x]
        log_phi_p = log_phi_all[idx_p]
        output_phi_x = output_phi_all[idx_x]
        var_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - 1 * torch.mean(log_phi_p)

        # perform Mixup and calculate the regularization
        target_x = output_phi_x[:, 1].exp()
        target_p = torch.ones(len(data_p), dtype=torch.float32)
        target_p = target_p.cuda() if torch.cuda.is_available() else target_p
        rand_perm = torch.randperm(data_p.size(0))
        data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
        m = torch.distributions.beta.Beta(config.alpha, config.alpha)
        lam = m.sample()
        data = lam * data_x + (1 - lam) * data_p_perm
        target = lam * target_x + (1 - lam) * target_p_perm
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        out_log_phi_all, _ = model_phi(data)
        reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()

        # calculate gradients and update the network
        phi_loss = var_loss + config.lam * reg_mix_log
        # ablation study on mixup regularization
        # phi_loss = var_loss
        opt_phi.zero_grad()
        phi_loss.backward()
        opt_phi.step()
        # scheduler.step()
        # update the utilities for analysis of the model
        reg_avg.update(reg_mix_log.item())
        phi_loss_avg.update(phi_loss.item())
        var_loss_avg.update(var_loss.item())
        phi_p, phi_x = log_phi_p.exp(), log_phi_x.exp()
        phi_p_avg.update(phi_p.mean().item(), len(phi_p))
        phi_x_avg.update(phi_x.mean().item(), len(phi_x))
    return phi_loss_avg.avg, var_loss_avg.avg, reg_avg.avg, phi_p_avg.avg, phi_x_avg.avg

def evaluate(config, model_phi, x_loader, test_loader, val_p_loader, val_x_loader, epoch, phi_loss, var_loss, reg_loss):
    """
    evaluate the performance on test set, and calculate the variational loss on validation set.

    :param model_phi: current model \Phi
    :param x_loader: loader for the whole training set (positive and unlabeled).
    :param test_loader: loader for the test set (fully labeled).
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    :param epoch: current epoch.
    :param phi_loss: VPU loss of the current epoch, which equals to var_loss + reg_loss.
    :param var_loss: variational loss of the training set.
    :param reg_loss: regularization loss of the training set.
    """

    # set the model to evaluation mode
    model_phi.eval()

    # calculate variational loss of the validation set consisting of PU data
    val_var = cal_val_var(model_phi, val_p_loader, val_x_loader)

    # max_phi is needed for normalization
    log_max_phi = -math.inf
    for idx, (data, _) in enumerate(x_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        new_log_max_phi,_ = model_phi(data)

        log_max_phi = max(log_max_phi, new_log_max_phi[:, 1].max())


    # feed test set to the model and calculate accuracy and AUC
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            log_phi,_ = model_phi(data)
            

            log_phi = log_phi[:, 1]

            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))

    pred_all = np.array((log_phi_all > math.log(config.th)).cpu().detach())
    result = torch.pow(10, log_phi_all)
    
    log_phi_all = np.array(log_phi_all.cpu().detach())
    target_all = np.array(target_all.cpu().detach())
    test_acc = accuracy_score(target_all, pred_all)
    print('Train Epoch: {}\t phi_loss: {:.4f}   var_loss: {:.4f}   reg_loss: {:.4f}   Test accuracy: {:.4f}   Val var loss: {:.4f}' \
          .format(epoch, phi_loss, var_loss, reg_loss, test_acc, val_var))
    return val_var, test_acc,log_max_phi

def cal_val_var(model_phi, val_p_loader, val_x_loader):
    """
    Calculate variational loss on the validation set, which consists of only positive and unlabeled data.

    :param model_phi: current \Phi model.
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    """

    # set the model to evaluation mode
    model_phi.eval()

    # feed the validation set to the model and calculate variational loss
    with torch.no_grad():
        for idx, (data_x, _) in enumerate(val_x_loader):
            if torch.cuda.is_available():
                data_x = data_x.cuda()
            output_phi_x_curr,_ = model_phi(data_x)
            if idx == 0:
                output_phi_x = output_phi_x_curr
            else:
                output_phi_x = torch.cat((output_phi_x, output_phi_x_curr))
        for idx, (data_p, _) in enumerate(val_p_loader):
            if torch.cuda.is_available():
                data_p = data_p.cuda()
            output_phi_p_curr,_ = model_phi(data_p)
            if idx == 0:
                output_phi_p = output_phi_p_curr
            else:
                output_phi_p = torch.cat((output_phi_p, output_phi_p_curr))
        log_phi_p = output_phi_p[:, 1]
        log_phi_x = output_phi_x[:, 1]
        var_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - torch.mean(log_phi_p)
        return var_loss.item()

def get_vpu_label(config,log_max_phi):
    from dataset.dataset_urine import get_urine_loaders_inference
    train_loader, test_loader = get_urine_loaders_inference(config)
    
    # set the model to evaluation mode
    model_phi.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            log_phi,_ = model_phi(data)
            log_phi = log_phi[:, 1]

            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))

    pred_all = np.array((log_phi_all > math.log(config.th)).cpu().detach())
   

def ROC_curve(test_labels, test_score,save_name):
    fpr, tpr, thresholds = roc_curve(test_labels, test_score)
    print('AUC: {}'.format(auc(fpr, tpr)))
    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i / 20.0 for i in range(21)])
    plt.xticks([i / 20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('./result/' + save_name )
    
