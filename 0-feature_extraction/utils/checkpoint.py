from utils.func import *
import shutil
import torch


def get_checkpoint_path(config):
    """
    return the path of saving current model.
    """
    # checkpoint_path = os.path.join(os.getcwd(),'save', config.dataset+'P=' + str(config.positive_label_list)+
    #                                '_lr=' + str(config.lr)+'_lambda=' + str(config.lam)+
    #                                '_alpha=' + str(config.alpha))
    checkpoint_path = os.path.join('.', '_'.join((config.dataset,'lr='+str(config.lr), 'lambda='+str(config.lam), 'alpha='+str(config.alpha), 'scale='+str(config.scale))), 'fold='+str(config.nth_fold))
    return checkpoint_path


def save_checkpoint(state, is_lowest_on_val, is_highest_on_test, config, epoch):
    """
    Save the current model to the checkpoint_path

    :param state: information of the model and training.
    :param is_lowest_on_val: indicating whether the current model has the lowest KL divergence on the validation set.
    :param is_highest_on_test: indicating whether the current model has the highest test accuracy.
    :param config: arguments.
    :param filename: name of the file that saves the model.
    """
    filename = str(epoch)+'.pth'
    checkpoint = get_checkpoint_path(config)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint ,filename)
    torch.save(state, filepath)
    if is_lowest_on_val:
        shutil.copyfile(filepath, os.path.join(checkpoint , 'model_lowest_on_val.pth'))
    if is_highest_on_test:
        shutil.copyfile(filepath, os.path.join(checkpoint , '_model_highest_on_test.pth'))
