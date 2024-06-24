import glob
import numpy as np
import os

for nth_fold in range(1):
    print(nth_fold)
    if nth_fold > 0:
        graph_root1 = '/bigdata/projects/beidi/data/scale128_fea384_' + str(nth_fold)
        graph_root2 = '/bigdata/projects/beidi/data/scale256_fea768_' + str(nth_fold)
        
    else:
        graph_root1 ='/bigdata/projects/beidi/data/VPU/scale128_fea512'
        graph_root2 ='/bigdata/projects/beidi/data/VPU/scale256_fea1024_ep9_notaligned'
    for split in ['train','test']:
        for c in ['benign','atypical','suspicious','cancer']:
        # for c in ['B','M']:
            save_root = os.path.join('/bigdata/projects/beidi/data/multiscale_'+str(nth_fold),split,c)
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            print(os.path.join(graph_root1,split,c)+'/*')
            filelist1 = glob.glob(os.path.join(graph_root1,split,c)+'/*')
            filelist1 = sorted(filelist1)
            print(filelist1)
       
            filelist2 = glob.glob(os.path.join(graph_root2,split,c)+'/*')
            filelist2 = sorted(filelist2)
            # print(filelist2)
            for i in range(len(filelist1)):
                file_name = filelist1[i].split('/')[-1]
                print(file_name)
                embedding_small = np.load(filelist1[i])
                embedding_big = np.load(filelist2[i])
                print(embedding_small.shape)
                print(embedding_big.shape)

                # embedding = np.concatenate((embedding_small,embedding_big),axis = 1)

                # np.save(os.path.join(save_root,file_name),embedding)

