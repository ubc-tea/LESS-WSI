import argparse


def args_parser():

    ## ------------------------ For feature extraction ------------------------
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
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

    ## ------------------------ For aggregation ------------------------
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='crossvit_base_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=240, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--pretrained', action='store_true', help='load imagenet1k pretrained model')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--crop-ratio', type=float, default=256/224, help='crop ratio for evaluation')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/bigdata/projects/beidi/data', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='urine', choices=['urine','CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--no-resume-loss-scaler', action='store_false', dest='resume_loss_scaler')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='disable amp')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--auto-resume', action='store_true', help='auto resume')
    parser.add_argument('--finetune', action='store_true', help='finetune model')
    parser.add_argument('--initial_checkpoint', type=str, default='/bigdata/projects/beidi/git/crossvit/pretrain/crossvit_large_224.pth', help='path to the pretrained model')
    parser.add_argument('--nth_fold', type=int, default= 0)
    parser.add_argument('--VPUep', type=int, default=0)
    parser.add_argument('--vpu_dim', type=str, default='384_768')
    parser.add_argument('--features', type=str, default='VPU')

    return parser.parse_args()