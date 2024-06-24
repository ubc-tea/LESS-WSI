import argparse
import random
from vpu import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Pytorch Variational Positive Unlabeled Learning')
parser.add_argument('--dataset', default='urine',choices=['urine','FANC'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--val-iterations', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--num_labeled', type=int, default=3000, help="number of labeled positive samples")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.3, help="parameter in Mixup")
parser.add_argument('--lam', type=float, default=0.03, help="weight of the regularizer")
parser.add_argument('--th', type=float, default=0.5, help="threshold of decision")
parser.add_argument('--scale', type=int, default=128, help="scale")
parser.add_argument('--gnnlr', type=float, default=5e-4)
parser.add_argument('--gnnbs', type=int, default=1)
parser.add_argument('--get_feature', default= 1)
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--nth_fold',type=int, default=0)
parser.add_argument('--VPUep',type=int, default=10)
parser.add_argument('--save_dir',default= './save',help="the path to save pretrained VPU")
parser.add_argument('--slide_root',default='/bigdata/projects/beidi/data/tile256to128_rand100_new',help="the path of raw patches")
parser.add_argument('--feature_root',default='./saved_feature',help="the path to save features from pretrained VPU")
parser.add_argument('--get_label', action='store_false', help='Get VPU predicted labels')

args = parser.parse_args()

if args.dataset == 'urine':
    from model.model_urine_crossvit import NetworkPhi_scale128, NetworkPhi_scale256
    if args.scale == 128:
        NetworkPhi = NetworkPhi_scale128
    elif args.scale == 256:
        NetworkPhi = NetworkPhi_scale256
    from dataset.dataset_urine import get_urine_loaders as get_loaders
    parser.add_argument('--positive_label_list', type=list, default=[0])

elif args.dataset == 'FANC':
    from model.model_urine_crossvit import NetworkPhi
    from dataset.dataset_FANC import get_urine_loaders as get_loaders
    parser.add_argument('--positive_label_list', type=list, default=[0])

elif args.dataset == 'cifar10':
    from model.model_cifar import NetworkPhi
    from dataset.dataset_cifar import get_cifar10_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[0, 1, 8, 9])
elif args.dataset == 'fashionMNIST':
    from model.model_fashionmnist import NetworkPhi
    from dataset.dataset_fashionmnist import get_fashionMNIST_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[1, 4, 7])
elif args.dataset == 'stl10':
    from model.model_stl import NetworkPhi
    from dataset.dataset_stl import get_stl10_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[0, 2, 3, 8, 9])
elif args.dataset == 'pageblocks':
    from model.model_vec import NetworkPhi
    from dataset.dataset_pageblocks import get_pageblocks_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[2, 3, 4, 5])
elif args.dataset == 'grid':
    from model.model_vec import NetworkPhi
    from dataset.dataset_grid import get_grid_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=[1])
elif args.dataset == 'avila':
    from model.model_vec import NetworkPhi
    from dataset.dataset_avila import get_avila_loaders as get_loaders

    parser.add_argument('--positive_label_list', type=list, default=['A'])
else:
    assert False
args = parser.parse_args()

def get_mean_std(train_loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in train_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

def main(config,nth_fold):
    # set up cuda if it is available
    if torch.cuda.is_available():
        CUDA_VISIBLE_DEVICES = 0
        torch.cuda.set_device(config.gpu)

    # set up the loaders
    if config.dataset in ['urine','FANC']:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader = get_loaders(batch_size=config.batch_size,positive_label_list=config.positive_label_list,nth_fold=nth_fold)

    if config.dataset in ['cifar10', 'fashionMNIST', 'stl10']:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader, idx = get_loaders(batch_size=config.batch_size,
                                                                                       num_labeled=config.num_labeled,
                                                                                       positive_label_list=config.positive_label_list)
    elif config.dataset in ['avila', 'pageblocks', 'grid']:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader = get_loaders(batch_size=config.batch_size,
                                                                                  num_labeled=config.num_labeled,
                                                                                  positive_label_list=config.positive_label_list)
    loaders = (p_loader, x_loader, val_p_loader, val_x_loader, test_loader)
    # mean, std = get_mean_std(p_loader)

    # please read the following information to make sure it is running with the desired setting
    print('==> Preparing data')
    print('    # train data: ', len(x_loader.dataset))
    print('    # labeled train data: ', len(p_loader.dataset))
    print('    # test data: ', len(test_loader.dataset))
    print('    # val x data:', len(val_x_loader.dataset))
    print('    # val p data:', len(val_p_loader.dataset))

    # something about saving the model
    checkpoint = get_checkpoint_path(config)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    # call VPU
    run_vpu(config, loaders, NetworkPhi,nth_fold)

if __name__ == '__main__':
    
    setup_seed(args.seed)
    main(args,args.nth_fold)
