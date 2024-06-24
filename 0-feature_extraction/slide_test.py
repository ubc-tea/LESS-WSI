from torch_geometric.loader import DataLoader
from feature_construction import UrineDataset_for_graph, get_fea,UrineDataset_for_graph_multiscale
from dataset.dataset_urine import UrineSlideDataset, EmbedDataset
import os
import glob
import shutil
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import wandb
import torch.nn as nn
import math

'''=================================================================================================================='''
'''======================================================= GNN ======================================================'''
'''=================================================================================================================='''

def gcn_train(model,train_loader,criterion,optimizer,scheduler):
    correct = 0
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.

        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    optimizer.step()  # Update parameters based on gradients.
    scheduler.step()
    optimizer.zero_grad()  # Clear gradients.
    return correct / len(train_loader.dataset), loss

def gcn_test(model,loader,criterion,nth_fold):
    if nth_fold > 0:
        slide_root = '/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)
    else:
        slide_root = '/bigdata/projects/beidi/data/tile128_rand100_new'
    slide_test = os.path.join(slide_root, 'test')
    # slide_test = '/bigdata/projects/beidi/dataset/urine/2306/tile256'
    test_name = glob.glob(slide_test + '/*' + '/*')
    test_name = sorted(test_name)
    # print(len(test_name))
    correct = 0
    model.eval()
    targetlist = []
    predlist = []
    outputlist = []
    for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)  # Compute the loss.
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         # print('pred',pred)
         # print('y   ',data.y)
         # Confusion Matrix
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         predlist += pred.eq(data.y).cpu().numpy().tolist()
         outputlist += pred
         targetlist += data.y
    false_list_b = []
    false_list_a = []
    false_list_s = []
    false_list_c = []
    # print(len(predlist))
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
    print('false list:','b:',false_list_b,'a:',false_list_a,'s:',false_list_s,'c:',false_list_c)
    from sklearn.metrics import confusion_matrix
    print('Confusion Matrix:')
    confu_matrix = confusion_matrix(targetlist, outputlist)
    print(confu_matrix)
    if targetlist.count(1) != len(outputlist) or outputlist.count(0) != len(outputlist):
        AUC = roc_auc_score(targetlist,outputlist)
    else:
        AUC = 0
    # AUC = roc_auc_score(data.y.tolist(), pred.tolist())
    Recall =  recall_score(targetlist, outputlist)
    Precision = precision_score(targetlist, outputlist)
    F1 = f1_score(targetlist,outputlist)
    Sensitivity =  confu_matrix[0,0]/(confu_matrix[0,0]+confu_matrix[1,0])
    Specificity = confu_matrix[1,1]/(confu_matrix[0,1]+confu_matrix[1,1])
    # print('AUC:', AUC)
    # print('Recall:',Recall)
    # print('Precision:', Precision)
    # print('F1 score:', F1)
    # print('Sensitivity:', confu_matrix[0,0]/(confu_matrix[0,0]+confu_matrix[1,0]))
    # print('Specificity:', confu_matrix[1,1]/(confu_matrix[0,1]+confu_matrix[1,1]))
    # ROC
    # ROC_curve(data.y.tolist(), pred.tolist(), save_name='/slide_roc_GNN')
    return correct / len(loader.dataset), loss, model, AUC,Recall, Precision , F1, Sensitivity, Specificity # Derive ratio of correct predictions.

def Graph_classification(config,model_phi,nth_fold,epoch): 
     
    graph_root = os.path.join('.','saved_feature','scale' + str(config.scale), str(nth_fold))
    graph_train = os.path.join(graph_root, 'train')
    graph_test = os.path.join(graph_root, 'test')
    train_dataset = UrineDataset_for_graph(root=graph_train)
    test_dataset = UrineDataset_for_graph(root=graph_test)
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=config.gnnbs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.gnnbs, shuffle=False)
    
    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.gnnlr)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40,eta_min=1e-4)
    test_best = 0
    count = 0
    acc = []
    
    watermark = "bs{}_lr{}_seed{}_fold{}".format(config.gnnbs, config.gnnlr,config.seed,nth_fold)
    wandb.init(project="GNN",
                group = 'VPU256_multi_cos_1layer',
               name=watermark)
    wandb.config.update(config)
    
    for epoch in range(40):
        train_acc, train_loss = gcn_train(model, train_loader, criterion, optimizer,scheduler)
        # train_acc, train_loss = gcn_test(model, train_loader, criterion)
        test_acc, test_loss,GNN_model,AUC,Recall,Precision ,F1,Sensitivity,Specificity = gcn_test(model, test_loader, criterion,nth_fold=nth_fold)
        acc.append(test_acc)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': test_loss, "val_acc": test_acc,
                   "lr": optimizer.param_groups[0]["lr"],'AUC':AUC, 'Recall':Recall,'Precision':Precision,'F1':F1,'Sensitivity':Sensitivity,'Specificity':Specificity})
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f},'
              f'AUC: {AUC:.4f},Recall: {Recall:.4f},Precision: {Precision:.4f} ,F1: {F1:.4f},Sensitivity: {Sensitivity:.4f},Specificity: {Specificity:.4f}')
        if epoch == 0:
            test_best_acc = test_acc
        if test_acc <= test_best_acc:
            test_best_acc = test_acc
        print('Max Acc and epoch',max(acc),acc.index(max(acc)))
    return GNN_model, test_best_acc

'''=================================================================================================================='''
'''==================================================Multi GNN ======================================================'''
'''=================================================================================================================='''
def Graph_classification_multiscale(config,model_phi,model,nth_fold):

    if nth_fold > 0:
        slide_root1 = '/bigdata/projects/beidi/data/tile256_rand100_new_kfold/'+str(nth_fold)
        graph_root1 = '/bigdata/projects/beidi/data/scale128_' + str(nth_fold)
        slide_root2 = '/bigdata/projects/beidi/data/tile256to256_rand100_new_kfold/'+str(nth_fold)
        graph_root2 = '/bigdata/projects/beidi/data/scale128_' + str(nth_fold)
    else:
        slide_root1 = '/bigdata/projects/beidi/data/tile256_rand100_new'
        graph_root1 = '/bigdata/projects/beidi/data/scale128'
        slide_root2 = '/bigdata/projects/beidi/data/tile256to256_rand100_new'
        graph_root2 = '/bigdata/projects/beidi/data/scale128'

    slide_train1 = os.path.join(slide_root1, 'train')
    slide_test1 = os.path.join(slide_root1, 'test')
    graph_train1 = os.path.join(graph_root1, 'train')
    graph_test1 = os.path.join(graph_root1, 'test')
    slide_train2 = os.path.join(slide_root2, 'train')
    slide_test2 = os.path.join(slide_root2, 'test')
    graph_train2 = os.path.join(graph_root2, 'train')
    graph_test2 = os.path.join(graph_root2, 'test')

    train_name = glob.glob(slide_train1 + '/*' + '/*')
    train_name = sorted(train_name)
    test_name = glob.glob(slide_test1 + '/*' + '/*')
    test_name = sorted(test_name)

    train_dataset = UrineDataset_for_graph_multiscale(root1=graph_train1,root2=graph_train2)
    test_dataset = UrineDataset_for_graph_multiscale(root1=graph_test1,root2=graph_test2)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=config.gnnbs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.gnnbs, shuffle=False)

    print(len(train_loader))
    # assert 2 == 3
    # model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.gnnlr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40,eta_min=1e-4)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    test_best = 0
    count = 0
    acc = []

    watermark = "bs{}_lr{}_seed{}_fold{}".format(config.gnnbs, config.gnnlr,config.seed,nth_fold)
    wandb.init(project="VPU_GNN_muitl",
               name=watermark)
    wandb.config.update(config)

    for epoch in range(40):
        train_acc, train_loss = gcn_train(model, train_loader, criterion, optimizer,scheduler)
        # train_acc, train_loss = gcn_test(model, train_loader, criterion)
        test_acc, test_loss,GNN_model,AUC,Recall,Precision ,F1,Sensitivity,Specificity = gcn_test(model, test_loader,criterion,nth_fold=nth_fold)
        acc.append(test_acc)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': test_loss, "val_acc": test_acc,
                   "lr": optimizer.param_groups[0]["lr"],'AUC':AUC, 'Recall':Recall,'Precision':Precision,'F1':F1,'Sensitivity':Sensitivity,'Specificity':Specificity})
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f},'
              f'AUC: {AUC:.4f},Recall: {Recall:.4f},Precision: {Precision:.4f} ,F1: {F1:.4f},Sensitivity: {Sensitivity:.4f},Specificity: {Specificity:.4f}')
        if epoch == 0:
            test_best_acc = test_acc
        if test_acc <= test_best_acc:
            test_best_acc = test_acc
        print('Max Acc and epoch',max(acc),acc.index(max(acc)))
    return GNN_model, test_best_acc




'''=================================================================================================================='''
'''=================================================== Threshold ===================================================='''
'''=================================================================================================================='''
def Threshold(model,args,epoch):
    watermark = "seed{}_fold{}_vpuTh{}".format( args.seed, args.nth_fold,args.th)
    wandb.init(project="VPU+threshold256_0304",
               name=watermark)
    wandb.config.update(args, allow_val_change=True)
    # slide-level train
    import glob
    # rootDir = args.val_dataset_path
    if args.nth_fold > 0:
        rootDir = '/bigdata/projects/beidi/data/tile256_rand100_new_kfold/'+str(args.nth_fold )+'/test'
    else:
        rootDir = '/bigdata/projects/beidi/data/tile256_rand100_new/test'
    name = []
    name = glob.glob(rootDir + '/*' + '/*')
    name = sorted(name)
    index_list = list(range(len(name)))
    # model_cell = VGGNet()
    # checkpoint = torch.load(args.saved_model_name)
    # model_cell.load_state_dict(checkpoint, strict=False)
    # model_slide = MLPclassifica()
    # summary(model_cell, input_size=(3, 32, 32), device='cpu')
    # summary(model_slide, input_size=(1000,50), device='cpu')
    # model_cell.eval()  # 模型进入测试阶段，参数不再更改
    # model_slide.eval()  # 模型进入测试阶段，参数不再更改
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("has GPU %d " % torch.cuda.device_count())
    #     model_cell = nn.DataParallel(model_cell, device_ids=[0, 1])
    #     model_slide = nn.DataParallel(model_slide, device_ids=[0, 1])
    label_list = []
    pred_list = []
    num_correct = 0
    count_cancer, count_benign, count_atypical, count_suspicious = 0, 0, 0, 0
    num_correct_cancer, num_correct_benign, num_correct_atypical, num_correct_suspicious = 0, 0, 0, 0
    for i in name:
        print('Slide',i.split('/')[-1],'is', i.split('/')[-2])
        file_path = os.path.join(rootDir,i)
        test_dataset = UrineSlideDataset(dataset_path=file_path)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # model_cell = model_cell.to(device)
        # model_slide = model_slide.to(device)
        # loss_func = nn.CrossEntropyLoss()

        if i.split('/')[-2] == 'benign':
            label = 0
            label_list.append(0)
            count_benign += 1
        elif i.split('/')[-2] == 'atypical':
            label = 0
            label_list.append(0)
            count_atypical += 1
        elif i.split('/')[-2] == 'cancer':
            label = 1
            label_list.append(1)
            count_cancer += 1
        elif i.split('/')[-2] == 'suspicious':
            label = 1
            label_list.append(1)
            count_suspicious += 1
        else:
            assert 2 == 3
        # get cell embedding
        output = torch.empty(0,2).cuda().detach()
        # output = torch.empty(0,4).to(device).detach()
        for data in test_loader:  # 测试模型
            img, _ = data

            img = img.cuda().detach() # 测试时不需要梯度
            # _,out = model_cell(img)
            out,_ = model(img)
            output = torch.cat([output, out], dim=0, out=None)
            # print(out)
        # _, pred = torch.max(output, 1)
        pred = output[:, 1]
        pred = (pred < math.log(args.th)).cpu().detach()
        count_cancer_cell = pred.sum()
        pred_list.append((count_cancer_cell.item())>50)
        print('count cancer cell {:.0f}'.format(count_cancer_cell))
        if count_cancer_cell > 50: ## adjust threshold
            slide_pred = 1
        else:
            slide_pred = 0

        if slide_pred == label:
            num_correct += 1
            if i.split('/')[-2] == 'benign':
                num_correct_benign += 1
            elif i.split('/')[-2] == 'atypical':
                num_correct_atypical += 1
            elif i.split('/')[-2] == 'cancer':
                num_correct_cancer += 1
            elif i.split('/')[-2] == 'suspicious':
                num_correct_suspicious += 1
            else:
                assert 2 == 3
    num_correct = num_correct_suspicious + num_correct_atypical + num_correct_cancer + num_correct_suspicious
    slide_acc = num_correct/len(name)
    print('Slide level acc is {:.4f}'.format(slide_acc))
    print('Acc of benign is {:.4f}'.format(num_correct_benign/count_benign))
    print('Acc of atypical is {:.4f}'.format(num_correct_atypical/count_atypical))
    print('Acc of suspicious is {:.4f}'.format(num_correct_suspicious/count_suspicious))
    print('Acc of cancer is {:.4f}'.format(num_correct_cancer/count_cancer))
    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    # print(label_list)
    # print(pred_list)
    print('Confusion Matrix:')
    confu_matrix = confusion_matrix(label_list, pred_list)
    print(confu_matrix)
    AUC = roc_auc_score(label_list, pred_list)
    Recall =  recall_score(label_list, pred_list)
    Precision = precision_score(label_list, pred_list)
    F1 = f1_score(label_list, pred_list)
    Sensitivity =  confu_matrix[0,0]/(confu_matrix[0,0]+confu_matrix[1,0])
    Specificity = confu_matrix[1,1]/(confu_matrix[0,1]+confu_matrix[1,1])
    wandb.log({'epoch': epoch, "slide_acc": slide_acc,
                'AUC': AUC, 'Recall': Recall, 'Precision': Precision, 'F1': F1,
               'Sensitivity': Sensitivity, 'Specificity': Specificity})



'''=================================================================================================================='''
'''==================================================== MLP ========================================================='''
'''=================================================================================================================='''
from MLP_model import MLP
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import explained_variance_score, mean_squared_error,r2_score

def train_one_epoch(model, loss_func, epoch, epoch_size, epoch_size_val, gen, gen_val, Full_Epoch,optimizer,scheduler):
    train_loss = 0
    val_loss = 0
    total_loss = 0
    total_val_loss = 0

    with tqdm(total=epoch_size, desc=f'Epoch{epoch + 1}/{Full_Epoch}', postfix=dict, mininterval=0.3) as pbar:
        correct = 0
        for iteration, (inputs, targets) in enumerate(gen):
            data, target = inputs.cuda(), targets.cuda()
            target= target.squeeze(dim=1).long()
            # for bstcn in gen:
            if iteration >= epoch_size_val:
                break

            optimizer.zero_grad()
            output = model(data)


            loss = loss_func(output, target)
            pred = output.argmax(dim=1)

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            train_loss = total_loss / (iteration + 1)
            # train_acc = r2_score(target.detach().numpy(), output.detach().numpy())

            correct += int((pred == target).sum())  # Check against ground-truth labels.

            # pbar.set_postfix(**{"total_loss": train_loss,
            #                     "learning_rate:": 1e-3,
            #                     "Acc": train_acc})
            # pbar.update(1)  # 更新进度条
        train_acc = correct / len(data)

    print(f"Epoch: {epoch + 1}")
    with tqdm(total=epoch_size_val, desc=f'Epoch{epoch + 1}/{Full_Epoch}', postfix=dict, mininterval=0.3) as pbar:
        correct = 0
        # for iteration, batch in enumerate(gen_val):
        for iteration, (inputs, targets) in enumerate(gen):
            data, target = inputs.cuda(), targets.cuda()
            target = target.squeeze(dim=1).long()
        # for bstcn in gen:
            if iteration >= epoch_size_val:
                break

            optimizer.zero_grad()
            output = model(data)


            loss = loss_func(output, target)
            pred = output.argmax(dim=1)
            total_val_loss += loss.item()
            val_loss = total_val_loss / (iteration + 1)
            # val_acc = r2_score(target.detach().numpy(), output.detach().numpy())
            correct += int((pred == target).sum())  # Check against ground-truth labels.
            val_acc = correct / len(gen_val)
            # pbar.set_postfix(**{"val_loss": val_loss,
            #                     "Acc": val_acc})
            # pbar.upd .ate(1)
        val_acc = correct / len(data)
        from sklearn.metrics import confusion_matrix
        print('Confusion Matrix:')
        target = target.cpu()
        pred = pred.cpu()
        confu_matrix = confusion_matrix(target, pred.tolist())
        print(confu_matrix)
        AUC = roc_auc_score(target, pred.tolist())
        Recall = recall_score(target, pred.tolist())
        Precision = precision_score(target, pred.tolist())
        F1 = f1_score(target, pred.tolist())
        Sensitivity = confu_matrix[0, 0] / (confu_matrix[0, 0] + confu_matrix[1, 0])
        Specificity = confu_matrix[1, 1] / (confu_matrix[0, 1] + confu_matrix[1, 1])
        print('Acc',val_acc,'AUC:', AUC,'Recall:',Recall,'Precision:', Precision,'F1 score:', F1,'Sensitivity:', confu_matrix[0,0]/(confu_matrix[0,0]+confu_matrix[1,0]),
              'Specificity:', confu_matrix[1,1]/(confu_matrix[0,1]+confu_matrix[1,1]))
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, "val_acc": val_acc,
                   "lr": optimizer.param_groups[0]["lr"], 'AUC': AUC, 'Recall': Recall, 'Precision': Precision,
                   'F1': F1, 'Sensitivity': Sensitivity, 'Specificity': Specificity})

    if epoch + 1 == Full_Epoch:
        torch.save(model.state_dict(),
                   'weights/mlp_weights-epoch%d-Total_loss%.4f-val_loss%.4f.pkl' % (
                   (epoch + 1), train_loss, val_loss / (iteration + 1)))

    return train_loss, train_acc, val_loss, val_acc

def MLP_main(args):
    Full_Epoch = 40
    Batch_size = 64
    lr = 2e-2
    watermark = "seed{}_fold{}_lr{}".format(args.seed, args.nth_fold,lr)
    wandb.init(project="VPU+MLP256",
               name=watermark)
    wandb.config.update(args, allow_val_change=True)

    # slide-level train


    if args.nth_fold == 0:
        root_train = '/bigdata/projects/beidi/data/scale128/train'
        root_test = '/bigdata/projects/beidi/data/scale128/test'
    else:
        root_train = '/bigdata/projects/beidi/data/scale128_' + str(args.nth_fold) + '/train'
        root_test = '/bigdata/projects/beidi/data/scale128_' + str(args.nth_fold) + '/test'
    train_dataset = EmbedDataset(dataset_path=root_train)
    gen = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=8)
    test_dataset = EmbedDataset(dataset_path=root_test)
    gen_val = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=8)

    loss_and_acc_curve = True

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    #
    # x_train_s, x_test_s, y_train, y_test = dataset()
    # x_train_st = torch.from_numpy(x_train_s.astype(np.float32))
    # x_test_st = torch.from_numpy(x_test_s.astype(np.float32))
    # y_train_t = torch.from_numpy(y_train.astype(np.float32))
    # y_test_t = torch.from_numpy(y_test.astype(np.float32))

    # train_dataset = TensorDataset(x_train_st, y_train_t)
    # test_dataset = TensorDataset(x_test_st, y_test_t)
    # gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0,
    #                  pin_memory=True, drop_last=True, shuffle=True)
    # gen_val = DataLoader(test_dataset, batch_size=Batch_size, num_workers=0,
    #                      pin_memory=True, drop_last=True, shuffle=True)

    model = MLP().cuda()
    optimizer = optim.SGD(model.parameters(), lr)
    loss_func = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    # loss_func = nn.MSELoss()
    # print(len(train_dataset))
    epoch_size = len(train_dataset) // Batch_size
    epoch_size_val = len(test_dataset) // Batch_size

    for epoch in range(0, Full_Epoch):
        train_loss, train_acc, val_loss, val_acc = train_one_epoch(model, loss_func, epoch, epoch_size, epoch_size_val,
                                                                   gen, gen_val, Full_Epoch,optimizer,scheduler)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

