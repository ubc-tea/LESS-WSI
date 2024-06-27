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
    for i, data in enumerate(val_loader):
        if i > 0:
            break
        images, labels = data
        images = images.to(device)
        outputs, encoded = model(images)
        score, pred = torch.max(outputs, 1)
        pred = pred.detach().cpu()
        encoded_features = encoded.detach().cpu().numpy()
        total_pred.append(pred)
        total_embeddings.append(encoded_features)
        # print()
        # print('Acc:',pred.eq(torch.squeeze(labels)).sum().item()/len(torch.squeeze(labels).numpy().tolist()))
    total_pred = np.squeeze(np.row_stack(total_pred))
    total_embeddings = np.row_stack(total_embeddings)

    return total_embeddings,total_pred

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
   
    train_name = glob.glob(os.path.join(slide_train, '*','*'))
    train_name = sorted(train_name)
    test_name = glob.glob(os.path.join(slide_test, '*','*'))
    test_name = sorted(test_name)
    if config.get_feature:
        for i in train_name:
            print('saving the feature of', i)
            file_path = os.path.join(slide_train, i)
            train_dataset = UrineSlideDataset(dataset_path=file_path,config=config)
            if len(glob.glob(file_path))==0:
                continue
            train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=4)
            embeddings_train, pred = get_fea(train_loader, model_phi)
            if not os.path.exists(os.path.join(feature_train,'cancer')):
                os.makedirs(os.path.join(feature_train,'cancer'))
            if not os.path.exists(os.path.join(feature_train,'benign')):
                os.makedirs(os.path.join(feature_train, 'benign'))
            if not os.path.exists(os.path.join(feature_train, 'atypical')):
                os.makedirs(os.path.join(feature_train, 'atypical'))
            if not os.path.exists(os.path.join(feature_train, 'suspicious')):
                os.makedirs(os.path.join(feature_train, 'suspicious'))
            np.save(feature_train +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + '.npy', embeddings_train)
            # np.save(feature_train +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + 'label.npy', pred)

        for i in test_name:
            print('saving the feature of', i)
            file_path = os.path.join(slide_test, i)
            test_dataset = UrineSlideDataset(dataset_path=file_path,config=config)
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
            embeddings_test,pred = get_fea(test_loader, model_phi)
            if not os.path.exists(os.path.join(feature_test,'cancer')):
                os.makedirs(os.path.join(feature_test,'cancer'))
            if not os.path.exists(os.path.join(feature_test,'benign')):
                os.makedirs(os.path.join(feature_test, 'benign'))
            if not os.path.exists(os.path.join(feature_test, 'atypical')):
                os.makedirs(os.path.join(feature_test, 'atypical'))
            if not os.path.exists(os.path.join(feature_test, 'suspicious')):
                os.makedirs(os.path.join(feature_test, 'suspicious'))
            np.save(feature_test +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + '.npy', embeddings_test)
            # np.save(feature_test +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + 'label.npy', pred)
