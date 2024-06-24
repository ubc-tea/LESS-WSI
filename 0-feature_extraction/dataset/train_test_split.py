import os,glob
import random
import shutil
from shutil import copy2,copytree

# --for annotated cells--
# data_class = 'benign'
# # data_class = 'cancer'
# datadir_normal_bf_crop =  '/bigdata/projects/beidi/dataset/urine/slide_'+data_class+'/*.xml'
# datadir_normal = '/bigdata/projects/beidi/dataset/urine/annotated_'+data_class
# rootDir = '/bigdata/projects/beidi/dataset/urine/Urine_annotated/'

# # --for screened cells--
# data_class = 'benign'
# # data_class = 'cancer'
# data_class = 'suspicious'
# data_class = 'atypical'
# datadir_normal_bf_crop =  '/bigdata/projects/beidi/dataset/urine/filtered_cell_' + data_class +'/BD' + '*.png'
# datadir_normal_bf_crop =  '/bigdata/projects/beidi/dataset/urine/filtered_cell_' + data_class +'/BD*'
# datadir_normal = '/bigdata/projects/beidi/dataset/urine/filtered_cell_' + data_class
# rootDir = '/bigdata/projects/beidi/dataset/urine/Urine_divide'

# --for raw cells--
data_class = 'benign'
# data_class = 'cancer'
# data_class = 'suspicious'
# data_class = 'atypical'
datadir_normal_bf_crop =  '/bigdata/projects/beidi/git/vpu-tilt/data/' + data_class +'/BD' + '*.png'
datadir_normal = '/bigdata/projects/beidi/git/vpu-tilt/data/' + data_class
rootDir = '/bigdata/projects/beidi/git/vpu-tilt/data'


name=[]
for file in glob.glob(datadir_normal_bf_crop):
    # print(file)
    # --for annotated slides--
    # name.append(file.split('.')[0].split('/')[-1])
    # --for screened slides--
    name.append(file.split('.')[0].split('/')[-1].split('_')[0])
    # name.append(file.split('/')[-1])
    # assert 2==3

name = list(set(name))
index_list = list(range(len(name)))

# random.shuffle(index_list)

list_train = name[:int(len(name) * 0.7)]
list_test = name[int(len(name) * 0.7):]
print(list_train)
print(list_test)
assert 2==3

for i in list_train:
    trainDir = rootDir + '/test/' + data_class + '/' + i+'/' +data_class+'/'
    # trainDir = rootDir + '/train/' + data_class + '/' + i+'/' +'cancer'+'/'
    if not os.path.exists(trainDir):
        os.makedirs(trainDir)
    # file_path = datadir_normal +'/'+ i +'/'+data_class+'/*.png'
    file_path = datadir_normal +'/'+ i +'*.png'

    file_all = []
    file_list = glob.glob(file_path)
    # print(file_list)

    # random.shuffle(file_list)
    for file in file_list[:1000]:
        file_all.append(file)
    for fileName in file_all:
        copy2(fileName, trainDir)

for i in list_test:
    testDir = rootDir + '/test/' + data_class + '/' + i+'/' +data_class+'/'
    # testDir = rootDir + '/test/' + data_class + '/' + i+'/' +'cancer'+'/'
    if not os.path.exists(testDir):
        os.makedirs(testDir)
    file_path = datadir_normal +'/'+ i+'*.png'
    file_all = []
    file_list = glob.glob(file_path)
    random.shuffle(file_list)
    for file in file_list[:1000]:
        file_all.append(file)
    for fileName in file_all:
        copy2(fileName, testDir)


# num = 0
# for i in name:
#     trainDir = rootDir + '/train/' + data_class+'/' + i
#     if not os.path.exists(trainDir):
#
#         os.makedirs(trainDir)
#     testDir = rootDir + '/test/' + data_class + '/'+ i
#     if not os.path.exists(testDir):
#         os.makedirs(testDir)
#     file_path = datadir_normal +'/'+ i +'_*.png'
#     file_all = []
#     for file in glob.glob(file_path):
#         file_all.append(file)
#     # print(file_all)
#
#     for fileName in file_all:
#         # --for annotated slides--
#         # if num < len(name) * 0.8:
#         #     copy2(os.path.join(datadir_normal,fileName), trainDir)
#         # else:
#         #     copy2(os.path.join(datadir_normal,fileName), testDir)
#         # --for screened slides--
#         if num < len(name) * 0.8:
#             copy2(fileName, trainDir)
#         else:
#             copy2(fileName, testDir)
#     num += 1
#


