from torch_geometric.loader import DataLoader
from graph_construction import UrineDataset_for_graph, constructing_graph
from dataset.dataset_urine import UrineSlideDataset
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
import wandb


def gcn_train(model,train_loader,criterion,optimizer,scheduler):
    correct = 0
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()
        optimizer.zero_grad()  # Clear gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(train_loader.dataset), loss

def gcn_test(model,loader,criterion,nth_fold):
    if nth_fold > 0:
        slide_root = '/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)
    else:
        slide_root = '/bigdata/projects/beidi/data/tile128_rand100_new'
    slide_test = os.path.join(slide_root, 'test')
    test_name = glob.glob(slide_test + '/*' + '/*')
    test_name = sorted(test_name)
    correct = 0
    model.eval()
    predlist = []
    for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)  # Compute the loss.
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         # print('pred',pred)
         # print('y   ',data.y)
         # Confusion Matrix
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
         predlist += pred.eq(data.y).cpu().numpy().tolist()
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
    print('false list:','b:',false_list_b,'a:',false_list_a,'s:',false_list_s,'c:',false_list_c)
    from sklearn.metrics import confusion_matrix
    print('Confusion Matrix:')
    confu_matrix = confusion_matrix(data.y.tolist(), pred.tolist())
    print(confu_matrix)
    AUC = roc_auc_score(data.y.tolist(), pred.tolist())
    Recall =  recall_score(data.y.tolist(), pred.tolist())
    Precision = precision_score(data.y.tolist(), pred.tolist())
    F1 = f1_score(data.y.tolist(), pred.tolist())
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
    return correct / len(loader.dataset), loss, model,AUC,Recall,Precision ,F1,Sensitivity,Specificity # Derive ratio of correct predictions.

def Graph_classification(config,model_phi,model,nth_fold):

    if nth_fold > 0:
        slide_root = '/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)
        graph_root = '/bigdata/projects/beidi/data/scale128_' + str(nth_fold)
    else:
        slide_root = '/bigdata/projects/beidi/data/tile128_rand100_new'
        graph_root = '/bigdata/projects/beidi/data/scale128'

    slide_train = os.path.join(slide_root, 'train')
    slide_test = os.path.join(slide_root, 'test')
    graph_train = os.path.join(graph_root, 'train')
    graph_test = os.path.join(graph_root, 'test')
    train_name = glob.glob(slide_train + '/*' + '/*')
    test_name = glob.glob(slide_test + '/*' + '/*')

    if config.get_feature:
        if os.path.exists(graph_root):
            shutil.rmtree(graph_root)
        os.makedirs(graph_train)
        os.makedirs(graph_test)
        # plt.figure(figsize=(8, 8))
        for i in train_name:
            file_path = os.path.join(slide_train, i)
            train_dataset = UrineSlideDataset(dataset_path=file_path)
            train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=4)
            embeddings_train,pred = constructing_graph(train_loader, model_phi)
            if not os.path.exists(os.path.join(graph_train,'cancer')):
                os.makedirs(os.path.join(graph_train,'cancer'))
            if not os.path.exists(os.path.join(graph_train,'benign')):
                os.makedirs(os.path.join(graph_train, 'benign'))
            if not os.path.exists(os.path.join(graph_train, 'atypical')):
                os.makedirs(os.path.join(graph_train, 'atypical'))
            if not os.path.exists(os.path.join(graph_train, 'suspicious')):
                os.makedirs(os.path.join(graph_train, 'suspicious'))
            np.save(graph_train +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + '.npy', embeddings_train)
            # np.save(graph_train +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + 'label.npy', pred)

        for i in test_name:
            file_path = os.path.join(slide_test, i)
            test_dataset = UrineSlideDataset(dataset_path=file_path)
            test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
            embeddings_test,pred = constructing_graph(test_loader, model_phi)
            if not os.path.exists(os.path.join(graph_test,'cancer')):
                os.makedirs(os.path.join(graph_test,'cancer'))
            if not os.path.exists(os.path.join(graph_test,'benign')):
                os.makedirs(os.path.join(graph_test, 'benign'))
            if not os.path.exists(os.path.join(graph_test, 'atypical')):
                os.makedirs(os.path.join(graph_test, 'atypical'))
            if not os.path.exists(os.path.join(graph_test, 'suspicious')):
                os.makedirs(os.path.join(graph_test, 'suspicious'))
            np.save(graph_test +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + '.npy', embeddings_test)
            # np.save(graph_test +'/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + 'label.npy', pred)



    train_dataset = UrineDataset_for_graph(root=graph_train)
    test_dataset = UrineDataset_for_graph(root=graph_test)
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=config.gnnbs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.gnnbs, shuffle=False)

    # model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.gnnlr)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    test_best = 0
    count = 0
    acc = []

    watermark = "bs{}_lr{}_seed{}_fold{}".format(config.gnnbs, config.gnnlr,config.seed,nth_fold)
    wandb.init(project="gnn128_0303nignt",
               name=watermark)
    wandb.config.update(config)

    for epoch in range(100):
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


def Threshold(model,args,epoch):
    watermark = "seed{}_fold{}_vpuTh{}".format( args.seed, args.nth_fold,args.th)
    wandb.init(project="VPU+threshold128_0304",
               name=watermark)
    wandb.config.update(args, allow_val_change=True)
    # slide-level train
    import glob
    # rootDir = args.val_dataset_path
    if args.nth_fold > 0:
        rootDir = '/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(args.nth_fold )+'/test'
    else:
        rootDir = '/bigdata/projects/beidi/data/tile128_rand100_new/test'
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
