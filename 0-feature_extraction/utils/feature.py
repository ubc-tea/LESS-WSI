import os
import glob
from dataset.dataset_urine import UrineSlideDataset, EmbedDataset
def get_feature_urine(config,model_phi,nth_fold,epoch):
    slide_root = os.path.join(config.slide_root,'scale' + str(config.scale),str(config.nth_fold))
    feature_root =  os.path.join(config.feature_root,'scale' + str(config.scale),str(config.nth_fold))
    

    slide_train = os.path.join(slide_root, 'train')
    slide_test = os.path.join(slide_root, 'test')
    feature_train = os.path.join(feature_root, 'train')
    feature_test = os.path.join(feature_root, 'test')
    if not os.path.exists(feature_train):
        os.makedirs(feature_train)
    if not os.path.exists(feature_test):
        os.makedirs(feature_test)
   
    train_name = glob.glob(slide_train + '/*' + '/*')
    train_name = sorted(train_name)
    test_name = glob.glob(slide_test + '/*' + '/*')
    test_name = sorted(test_name)

    if config.get_feature:
        for i in train_name:
            print('saving the feature of', i)
            file_path = os.path.join(slide_train, i)
            train_dataset = UrineSlideDataset(dataset_path=file_path)
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
            test_dataset = UrineSlideDataset(dataset_path=file_path)
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
