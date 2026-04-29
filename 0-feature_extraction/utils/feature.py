import os
import glob
import torch
import numpy as np
from dataset.dataset_urine import UrineSlideDataset, EmbedDataset
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_fea(val_loader, model):
    model.eval()
    total_embeddings = []
    total_pred = []
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            outputs, encoded = model(images)
            score, pred = torch.max(outputs, 1)
            pred = pred.detach().cpu()
            encoded_features = encoded.detach().cpu().numpy()
            total_pred.append(pred)
            total_embeddings.append(encoded_features)
    if len(total_embeddings) == 0:
        # Empty slide directory: return zero-row arrays so the caller can
        # decide whether to skip or fail.
        return np.empty((0, 0)), np.empty((0,), dtype=np.int64)
    total_pred = np.squeeze(np.row_stack(total_pred))
    total_embeddings = np.row_stack(total_embeddings)

    return total_embeddings, total_pred

def get_feature_urine(config,model_phi,nth_fold,epoch):
    slide_root = os.path.join(config.slide_root,str(config.nth_fold))
    feature_root =  os.path.join(config.feature_root,'scale' + str(config.scale),str(config.nth_fold))


    slide_train = os.path.join(slide_root, 'train')
    slide_test = os.path.join(slide_root, 'test')
    feature_train = os.path.join(feature_root, 'train')
    feature_test = os.path.join(feature_root, 'test')
    if not os.path.exists(feature_train):
        os.makedirs(feature_train)
    if not os.path.exists(feature_test):
        os.makedirs(feature_test)

    # Pre-create the four class subdirs once per split, instead of inside
    # the per-slide loop.
    for cls in ('cancer', 'benign', 'atypical', 'suspicious'):
        os.makedirs(os.path.join(feature_train, cls), exist_ok=True)
        os.makedirs(os.path.join(feature_test, cls), exist_ok=True)

    # Patch-level batch size used while feeding patches through the VPU
    # encoder. Default is 128, which fits one slide's 100 patches in a
    # single batch but also handles slides with more patches without
    # silently truncating (see get_fea, which now iterates the full loader).
    feature_batch_size = getattr(config, 'feature_batch_size', 128)

    train_name = sorted(glob.glob(os.path.join(slide_train, '*', '*')))
    test_name = sorted(glob.glob(os.path.join(slide_test, '*', '*')))

    if config.get_feature:
        for slide_path in train_name:
            print('saving the feature of', slide_path)
            train_dataset = UrineSlideDataset(dataset_path=slide_path, config=config)
            if len(train_dataset) == 0:
                print(f'  skip: no patches found in {slide_path}')
                continue
            train_loader = DataLoader(
                train_dataset,
                batch_size=feature_batch_size,
                shuffle=False,
                num_workers=4,
            )
            embeddings_train, pred = get_fea(train_loader, model_phi)
            cls = os.path.basename(os.path.dirname(slide_path))
            slide_id = os.path.basename(slide_path)
            np.save(os.path.join(feature_train, cls, slide_id + '.npy'),
                    embeddings_train)

        for slide_path in test_name:
            print('saving the feature of', slide_path)
            test_dataset = UrineSlideDataset(dataset_path=slide_path, config=config)
            if len(test_dataset) == 0:
                print(f'  skip: no patches found in {slide_path}')
                continue
            test_loader = DataLoader(
                test_dataset,
                batch_size=feature_batch_size,
                shuffle=False,
                num_workers=4,
            )
            embeddings_test, pred = get_fea(test_loader, model_phi)
            cls = os.path.basename(os.path.dirname(slide_path))
            slide_id = os.path.basename(slide_path)
            np.save(os.path.join(feature_test, cls, slide_id + '.npy'),
                    embeddings_test)
