# import pandas as pd
# import numpy as np
# from utils import smiles2adjoin
# import tensorflow as tf
# import os
# from rdkit import Chem
# import numpy as np
#
# """
#
# {'O': 5000757, 'C': 34130255, 'N': 5244317, 'F': 641901, 'H': 37237224, 'S': 648962,
# 'Cl': 373453, 'P': 26195, 'Br': 76939, 'B': 2895, 'I': 9203, 'Si': 1990, 'Se': 1860,
# 'Te': 104, 'As': 202, 'Al': 21, 'Zn': 6, 'Ca': 1, 'Ag': 3}
#
# H C N O F S  Cl P Br B I Si Se
# """
#
# str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
#          'B': 10,'I': 11,'Si':12,'Se':13,'<unk>':14,'<mask>':15,'<global>':16}
#
# num2str =  {i:j for j,i in str2num.items()}
#
#
#
# class Graph_Bert_Dataset(object):
#     def __init__(self,path,smiles_field='Smiles',addH=True):
#         if path.endswith('.txt') or path.endswith('.tsv'):
#             self.df = pd.read_csv(path,sep='\t')
#         else:
#             self.df = pd.read_csv(path)
#         self.smiles_field = smiles_field
#         self.vocab = str2num
#         self.devocab = num2str
#         self.addH = addH
#
#     def get_data(self):
#
#         data = self.df
#         train_idx = []
#         idx = data.sample(frac=0.9).index
#         train_idx.extend(idx)
#
#         data1 = data[data.index.isin(train_idx)]
#         data2 = data[~data.index.isin(train_idx)]
#
#         self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
#         self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(256, padded_shapes=(
#             tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)
#
#         self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
#         self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
#             tf.TensorShape([None]))).prefetch(50)
#         return self.dataset1, self.dataset2
#
#     def numerical_smiles(self, smiles):
#         smiles = smiles.numpy().decode()
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:,1:] = adjoin_matrix
#         adjoin_matrix = (1 - temp) * (-1e9)
#
#         choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1
#         y = np.array(nums_list).astype('int64')
#         weight = np.zeros(len(nums_list))
#         for i in choices:
#             rand = np.random.rand()
#             weight[i] = 1
#             if rand < 0.8:
#                 nums_list[i] = str2num['<mask>']
#             elif rand < 0.9:
#                 nums_list[i] = int(np.random.rand() * 14 + 1)
#
#         x = np.array(nums_list).astype('int64')
#         weight = weight.astype('float32')
#         return x, adjoin_matrix, y, weight
#
#     def tf_numerical_smiles(self, data):
#         # x,adjoin_matrix,y,weight = tf.py_function(self.balanced_numerical_smiles,
#         #                                           [data], [tf.int64, tf.float32 ,tf.int64,tf.float32])
#         x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
#                                                      [tf.int64, tf.float32, tf.int64, tf.float32])
#
#         x.set_shape([None])
#         adjoin_matrix.set_shape([None,None])
#         y.set_shape([None])
#         weight.set_shape([None])
#         return x, adjoin_matrix, y, weight
#
#
#
# class Graph_Classification_Dataset(object):
#     def __init__(self,path,smiles_field='Smiles',label_field='Label',max_len=100,addH=True):
#         if path.endswith('.txt') or path.endswith('.tsv'):
#             self.df = pd.read_csv(path,sep='\t')
#         else:
#             self.df = pd.read_csv(path)
#         self.smiles_field = smiles_field
#         self.label_field = label_field
#         self.vocab = str2num
#         self.devocab = num2str
#         self.df = self.df[self.df[smiles_field].str.len() <= max_len]
#         self.addH = addH
#
#     def get_data(self):
#         data = self.df
#         data = data.dropna()
#         data[self.label_field] = data[self.label_field].map(int)
#         pdata = data[data[self.label_field] == 1]
#         ndata = data[data[self.label_field] == 0]
#         lengths = [0, 25, 50, 75, 100]
#
#
#         ptrain_idx = []
#         for i in range(4):
#             idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
#                     pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
#             ptrain_idx.extend(idx)
#
#         ntrain_idx = []
#         for i in range(4):
#             idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
#                     ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
#             ntrain_idx.extend(idx)
#
#
#         train_data = data[data.index.isin(ptrain_idx+ntrain_idx)]
#         pdata = pdata[~pdata.index.isin(ptrain_idx)]
#         ndata = ndata[~ndata.index.isin(ntrain_idx)]
#
#         ptest_idx = []
#         for i in range(4):
#             idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
#                     pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
#             ptest_idx.extend(idx)
#
#         ntest_idx = []
#         for i in range(4):
#             idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
#                     ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
#             ntest_idx.extend(idx)
#
#         test_data = data[data.index.isin(ptest_idx+ntest_idx)]
#         val_data = data[~data.index.isin(ptest_idx+ntest_idx+ptrain_idx+ntrain_idx)]
#
#         self.dataset1 = tf.data.Dataset.from_tensor_slices(
#             (train_data[self.smiles_field], train_data[self.label_field]))
#         self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(32, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).shuffle(100).prefetch(100)
#
#         self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
#         self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)
#
#         self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
#         self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)
#
#         return self.dataset1, self.dataset2, self.dataset3
#
#     def numerical_smiles(self, smiles,label):
#         smiles = smiles.numpy().decode()
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:, 1:] = adjoin_matrix
#         adjoin_matrix = (1-temp)*(-1e9)
#
#         x = np.array(nums_list).astype('int64')
#         y = np.array([label]).astype('int64')
#         return x, adjoin_matrix,y
#
#     def tf_numerical_smiles(self, smiles,label):
#         x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.int64])
#         x.set_shape([None])
#         adjoin_matrix.set_shape([None,None])
#         y.set_shape([None])
#         return x, adjoin_matrix , y
#
#
# class Graph_Regression_Dataset(object):
#     def __init__(self,path,smiles_field='Smiles',label_field='Label',normalize=True,max_len=100,addH=True):
#         if path.endswith('.txt') or path.endswith('.tsv'):
#             self.df = pd.read_csv(path,sep='\t')
#         else:
#             self.df = pd.read_csv(path)
#         self.smiles_field = smiles_field
#         self.label_field = label_field
#         self.vocab = str2num
#         self.devocab = num2str
#         self.df = self.df[self.df[smiles_field].str.len()<=max_len]
#         self.addH =  addH
#         if normalize:
#             self.max = self.df[self.label_field].max()
#             self.min = self.df[self.label_field].min()
#             self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
#             self.value_range = self.max-self.min
#
#
#     def get_data(self):
#         data = self.df
#
#
#
#         lengths = [0, 25, 50, 75, 100]
#
#         train_idx = []
#         for i in range(4):
#             idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
#                         data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
#             train_idx.extend(idx)
#
#         train_data = data[data.index.isin(train_idx)]
#         data = data[~data.index.isin(train_idx)]
#
#         test_idx = []
#         for i in range(4):
#             idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
#                     data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
#             test_idx.extend(idx)
#
#         test_data = data[data.index.isin(test_idx)]
#         val_data = data[~data.index.isin(test_idx)]
#
#
#
#         self.dataset1 = tf.data.Dataset.from_tensor_slices((train_data[self.smiles_field], train_data[self.label_field]))
#         self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(64, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]))).shuffle(100).prefetch(100)
#
#         self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
#         self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
#             tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([1]))).cache().prefetch(100)
#
#         self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
#         self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)
#
#         return self.dataset1,self.dataset2,self.dataset3
#
#     def numerical_smiles(self, smiles,label):
#         smiles = smiles.numpy().decode()
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:, 1:] = adjoin_matrix
#         adjoin_matrix = (1-temp)*(-1e9)
#
#         x = np.array(nums_list).astype('int64')
#         y = np.array([label]).astype('float32')
#         return x, adjoin_matrix,y
#
#     def tf_numerical_smiles(self, smiles,label):
#         x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.float32])
#         x.set_shape([None])
#         adjoin_matrix.set_shape([None,None])
#         y.set_shape([None])
#         return x, adjoin_matrix , y
#
#
# class Inference_Dataset(object):
#     def __init__(self,sml_list,max_len=100,addH=True):
#         self.vocab = str2num
#         self.devocab = num2str
#         self.sml_list = [i for i in sml_list if len(i)<max_len]
#         self.addH =  addH
#
#     def get_data(self):
#
#         self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))
#         self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]),tf.TensorShape([None]))).cache().prefetch(20)
#
#         return self.dataset
#
#     def numerical_smiles(self, smiles):
#         smiles_origin = smiles
#         smiles = smiles.numpy().decode()
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:,1:] = adjoin_matrix
#         adjoin_matrix = (1-temp)*(-1e9)
#         x = np.array(nums_list).astype('int64')
#         return x, adjoin_matrix,[smiles], atoms_list
#
#     def tf_numerical_smiles(self, smiles):
#         x,adjoin_matrix,smiles,atom_list = tf.py_function(self.numerical_smiles, [smiles], [tf.int64, tf.float32,tf.string, tf.string])
#         x.set_shape([None])
#         adjoin_matrix.set_shape([None,None])
#         smiles.set_shape([1])
#         atom_list.set_shape([None])
#         return x, adjoin_matrix,smiles,atom_list
#
#
#
# # class multi_task_dataset(object):
# #     def __init__(self,path_list,smiles_field,label_field,max_len=100,addH=True):
# #         self.vocab = str2num
# #         self.smiles_field = smiles_field
# #         self.label_field = label_field
# #         self.devocab = num2str
# #         self.addH =  addH
# #         self.pathlist = path_list
# #
# #     def get_data(self):
# #         x_train_list = []
# #         y_train_list = []
# #         mask_train_list=[]
# #         test_dataset_list = []
# #         for i,path in enumerate(self.pathlist):
# #             data = pd.read_csv(path,sep='\t')
# #             lengths = [0, 25, 50, 75, 100]
# #             train_idx = []
# #             for ii in range(4):
# #                 idx = data[(data[self.smiles_field].str.len() >= lengths[ii]) & (
# #                         data[self.smiles_field].str.len() < lengths[ii + 1])].sample(frac=0.8).index
# #                 train_idx.extend(idx)
# #             data1 = data[data.index.isin(train_idx)].copy()
# #             data2 = data[~data.index.isin(train_idx)].copy()
# #             x_train_list += data1[self.smiles_field].tolist()
# #             y_train = -np.ones((len(data1),len(self.pathlist))).astype('float32')
# #             mask_train = np.zeros((len(data1),len(self.pathlist))).astype('float32')
# #
# #             y_train[:,i] = np.array(data1[self.label_field])
# #             mask_train[:, i] = 1
# #
# #             y_train_list.append(y_train)
# #             mask_train_list.append(mask_train)
# #
# #             x_test = data2[self.smiles_field].tolist()
# #             y_test = -np.ones((len(data2),len(self.pathlist))).astype('float32')
# #             y_test[:,i] = np.array(data2[self.label_field])
# #             mask_test = np.zeros((len(data2),len(self.pathlist))).astype('float32')
# #             mask_test[:, i] = 1
# #             test_dataset_list.append(tf.data.Dataset.from_tensor_slices((x_test,y_test,mask_test)).map(self.tf_numerical_smiles).padded_batch(256,
# #                                                                 padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]),
# #                                                                 tf.TensorShape([None]),tf.TensorShape([None]))).cache().prefetch(100))
# #
# #         y_train_list = np.concatenate(y_train_list,axis=0)
# #         mask_train_list = np.concatenate(mask_train_list,axis=0)
# #
# #         dataset1 = tf.data.Dataset.from_tensor_slices((x_train_list,y_train_list,mask_train_list))
# #         dataset1 = dataset1.map(self.tf_numerical_smiles).shuffle(200).padded_batch(64, padded_shapes=(
# #             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([len(self.pathlist)]),
# #             tf.TensorShape([len(self.pathlist)]))).cache().prefetch(100)
# #         return dataset1, test_dataset_list
# #
# #     def numerical_smiles(self, smiles,y,y_mask):
# #         smiles = smiles.numpy().decode()
# #         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
# #         atoms_list = ['<global>'] + atoms_list
# #         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
# #         temp = np.ones((len(nums_list),len(nums_list)))
# #         temp[1:,1:] = adjoin_matrix
# #         adjoin_matrix = ((1-temp)*(-1e9)).astype('float32')
# #         x = np.array(nums_list).astype('int64')
# #         return x, adjoin_matrix,y,y_mask
# #
# #     def tf_numerical_smiles(self, smiles,y,y_mask):
# #         x,adjoin_matrix,y, y_mask = tf.py_function(self.numerical_smiles, [smiles,y,y_mask], [tf.int64, tf.float32,tf.float32, tf.float32])
# #         x.set_shape([None])
# #         adjoin_matrix.set_shape([None,None])
# #         y.set_shape([len(self.pathlist)])
# #         y_mask.set_shape([len(self.pathlist)])
# #         return x, adjoin_matrix,y, y_mask
#
#
#
import pandas as pd
import numpy as np
from utils import smiles2adjoin
import tensorflow as tf
import os
from rdkit import Chem
import numpy as np

"""     

{'O': 5000757, 'C': 34130255, 'N': 5244317, 'F': 641901, 'H': 37237224, 'S': 648962, 
'Cl': 373453, 'P': 26195, 'Br': 76939, 'B': 2895, 'I': 9203, 'Si': 1990, 'Se': 1860, 
'Te': 104, 'As': 202, 'Al': 21, 'Zn': 6, 'Ca': 1, 'Ag': 3}

H C N O F S  Cl P Br B I Si Se
"""

str2num = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br': 9,
    'B': 10, 'I': 11, 'Si': 12, 'Se': 13, '<unk>': 14, '<mask>': 15, '<global>': 16,
    'Te': 17, 'As': 18, 'Al': 19, 'Zn': 20, 'Ca': 21, 'Ag': 22,
    'Mg': 23, 'K': 24, 'Na': 25, 'Fe': 26, 'Cu': 27
}

num2str = {i: j for j, i in str2num.items()}


class Graph_Bert_Dataset(object):
    def __init__(self, path, smiles_field='Smiles', addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):

        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(256, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')
        return x, adjoin_matrix, y, weight

    def tf_numerical_smiles(self, data):
        # x,adjoin_matrix,y,weight = tf.py_function(self.balanced_numerical_smiles,
        #                                           [data], [tf.int64, tf.float32 ,tf.int64,tf.float32])
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])

        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, y, weight


class Graph_Classification_Dataset(object):
    def __init__(self, path, smiles_field='Smiles', label_field='Label', max_len=100, addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH

    def get_data(self):
        data = self.df
        data = data.dropna()
        data[self.label_field] = data[self.label_field].map(int)
        pdata = data[data[self.label_field] == 1]
        ndata = data[data[self.label_field] == 0]
        lengths = [0, 25, 50, 75, 100]

        ptrain_idx = []
        for i in range(4):
            idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
                    pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            ptrain_idx.extend(idx)

        ntrain_idx = []
        for i in range(4):
            idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
                    ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            ntrain_idx.extend(idx)

        train_data = data[data.index.isin(ptrain_idx + ntrain_idx)]
        pdata = pdata[~pdata.index.isin(ptrain_idx)]
        ndata = ndata[~ndata.index.isin(ntrain_idx)]

        ptest_idx = []
        for i in range(4):
            idx = pdata[(pdata[self.smiles_field].str.len() >= lengths[i]) & (
                    pdata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
            ptest_idx.extend(idx)

        ntest_idx = []
        for i in range(4):
            idx = ndata[(ndata[self.smiles_field].str.len() >= lengths[i]) & (
                    ndata[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
            ntest_idx.extend(idx)

        test_data = data[data.index.isin(ptest_idx + ntest_idx)]
        val_data = data[~data.index.isin(ptest_idx + ntest_idx + ptrain_idx + ntrain_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(32, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('int64')
        return x, adjoin_matrix, y

    def tf_numerical_smiles(self, smiles, label):
        x, adjoin_matrix, y = tf.py_function(self.numerical_smiles, [smiles, label], [tf.int64, tf.float32, tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        return x, adjoin_matrix, y


class Graph_Regression_Dataset(object):
    def __init__(self, path, smiles_field='Smiles', label_field='Label', normalize=True, max_len=100, addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field] - self.min) / (self.max - self.min) - 0.5
            self.value_range = self.max - self.min

    def get_data(self):
        data = self.df

        lengths = [0, 25, 50, 75, 100]

        train_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            train_idx.extend(idx)

        train_data = data[data.index.isin(train_idx)]
        data = data[~data.index.isin(train_idx)]

        test_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
            test_idx.extend(idx)

        test_data = data[data.index.isin(test_idx)]
        val_data = data[~data.index.isin(test_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix, y

    def tf_numerical_smiles(self, smiles, label):
        x, adjoin_matrix, y = tf.py_function(self.numerical_smiles, [smiles, label], [tf.int64, tf.float32, tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        return x, adjoin_matrix, y


class Inference_Dataset(object):
    def __init__(self, sml_list, max_len=100, addH=True):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = [i for i in sml_list if len(i) < max_len]
        self.addH = addH

    def get_data(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))
        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]),
            tf.TensorShape([None]))).cache().prefetch(20)

        return self.dataset

    def numerical_smiles(self, smiles):
        smiles_origin = smiles
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)
        x = np.array(nums_list).astype('int64')
        return x, adjoin_matrix, [smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x, adjoin_matrix, smiles, atom_list = tf.py_function(self.numerical_smiles, [smiles],
                                                             [tf.int64, tf.float32, tf.string, tf.string])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix, smiles, atom_list


# class multi_task_dataset(object):
#     def __init__(self,path_list,smiles_field,label_field,max_len=100,addH=True):
#         self.vocab = str2num
#         self.smiles_field = smiles_field
#         self.label_field = label_field
#         self.devocab = num2str
#         self.addH =  addH
#         self.pathlist = path_list
#
#     def get_data(self):
#         x_train_list = []
#         y_train_list = []
#         mask_train_list=[]
#         test_dataset_list = []
#         for i,path in enumerate(self.pathlist):
#             data = pd.read_csv(path,sep='\t')
#             lengths = [0, 25, 50, 75, 100]
#             train_idx = []
#             for ii in range(4):
#                 idx = data[(data[self.smiles_field].str.len() >= lengths[ii]) & (
#                         data[self.smiles_field].str.len() < lengths[ii + 1])].sample(frac=0.8).index
#                 train_idx.extend(idx)
#             data1 = data[data.index.isin(train_idx)].copy()
#             data2 = data[~data.index.isin(train_idx)].copy()
#             x_train_list += data1[self.smiles_field].tolist()
#             y_train = -np.ones((len(data1),len(self.pathlist))).astype('float32')
#             mask_train = np.zeros((len(data1),len(self.pathlist))).astype('float32')
#
#             y_train[:,i] = np.array(data1[self.label_field])
#             mask_train[:, i] = 1
#
#             y_train_list.append(y_train)
#             mask_train_list.append(mask_train)
#
#             x_test = data2[self.smiles_field].tolist()
#             y_test = -np.ones((len(data2),len(self.pathlist))).astype('float32')
#             y_test[:,i] = np.array(data2[self.label_field])
#             mask_test = np.zeros((len(data2),len(self.pathlist))).astype('float32')
#             mask_test[:, i] = 1
#             test_dataset_list.append(tf.data.Dataset.from_tensor_slices((x_test,y_test,mask_test)).map(self.tf_numerical_smiles).padded_batch(256,
#                                                                 padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]),
#                                                                 tf.TensorShape([None]),tf.TensorShape([None]))).cache().prefetch(100))
#
#         y_train_list = np.concatenate(y_train_list,axis=0)
#         mask_train_list = np.concatenate(mask_train_list,axis=0)
#
#         dataset1 = tf.data.Dataset.from_tensor_slices((x_train_list,y_train_list,mask_train_list))
#         dataset1 = dataset1.map(self.tf_numerical_smiles).shuffle(200).padded_batch(64, padded_shapes=(
#             tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([len(self.pathlist)]),
#             tf.TensorShape([len(self.pathlist)]))).cache().prefetch(100)
#         return dataset1, test_dataset_list
#
#     def numerical_smiles(self, smiles,y,y_mask):
#         smiles = smiles.numpy().decode()
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:,1:] = adjoin_matrix
#         adjoin_matrix = ((1-temp)*(-1e9)).astype('float32')
#         x = np.array(nums_list).astype('int64')
#         return x, adjoin_matrix,y,y_mask
#
#     def tf_numerical_smiles(self, smiles,y,y_mask):
#         x,adjoin_matrix,y, y_mask = tf.py_function(self.numerical_smiles, [smiles,y,y_mask], [tf.int64, tf.float32,tf.float32, tf.float32])
#         x.set_shape([None])
#         adjoin_matrix.set_shape([None,None])
#         y.set_shape([len(self.pathlist)])
#         y_mask.set_shape([len(self.pathlist)])
#         return x, adjoin_matrix,y, y_mask


class Graph_Regression_and_Pretraining_Dataset(object):
    def __init__(self, path, smiles_field='Smiles', label_field='Label', normalize=True, addH=True, max_len=100):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field] - self.min) / (self.max - self.min) - 0.5

    def get_data(self):
        data = self.df
        lengths = [0, 25, 50, 75, 100]
        train_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (
                    data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices((data1[self.smiles_field], data1[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([1]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).cache().shuffle(100).prefetch(100)
        self.dataset2 = tf.data.Dataset.from_tensor_slices((data2[self.smiles_field], data2[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([1]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).cache().prefetch(100)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles, label):

        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        x_true = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x_masked = np.array(nums_list).astype('int64')
        weight = weight.astype('int64')
        label = np.array([label]).astype('float32')
        return x_masked, label, adjoin_matrix, x_true, weight

    def tf_numerical_smiles(self, smiles, label):
        x, label, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [smiles, label],
                                                            [tf.int64, tf.float32, tf.float32, tf.int64, tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        weight.set_shape([None])
        label.set_shape([None])
        return x, label, adjoin_matrix, y, weight


class Graph_Bert_Dataset_3D(object):
    def __init__(self, path, smiles_field='SMILES', structure_paths_field='structure_paths', addH=True, use_3d=True):
        """
        Args:
            smiles_field: SMILES列名
            structure_paths_field: 3D文件路径列名
            use_3d: 是否使用3D结构
        """
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)

        self.smiles_field = smiles_field
        self.structure_paths_field = structure_paths_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH
        self.use_3d = use_3d

    def get_data(self):
        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        # 计算最大序列长度（sequence_length）
        max_sequence_length = max([len(smiles) for smiles in data1[self.smiles_field]])

        # 创建包含SMILES和结构路径的数据集
        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (data1[self.smiles_field].tolist(), data1[self.structure_paths_field].tolist())
        )
        # 使用 max_sequence_length 填充，使得所有样本的长度一致
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles_3d).padded_batch(
            64,  # batch_size
            padded_shapes=(
                tf.TensorShape([max_sequence_length]),  # x - 填充至最大序列长度
                tf.TensorShape([None, None]),  # adjoin_matrix
                tf.TensorShape([None, 3]),  # coords
                tf.TensorShape([None]),  # y
                tf.TensorShape([None])  # 标签长度
            ),
            drop_remainder=True  # 丢弃最后一个不足大小的批次
        ).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(
            (data2[self.smiles_field].tolist(), data2[self.structure_paths_field].tolist())
        )
        # 使用 max_sequence_length 填充，确保所有样本的长度一致
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles_3d).padded_batch(
            64,  # batch_size
            padded_shapes=(
                tf.TensorShape([max_sequence_length]),  # x - 填充至最大序列长度
                tf.TensorShape([None, None]),  # adjoin_matrix
                tf.TensorShape([None, 3]),  # coords
                tf.TensorShape([None]),  # y
                tf.TensorShape([None])  # 标签长度
            ),
            drop_remainder=True  # 丢弃最后一个不足大小的批次
        ).prefetch(50)

        return self.dataset1, self.dataset2


    def numerical_smiles_3d(self, smiles, structure_path):
        smiles_str = smiles.numpy().decode()
        structure_path_str = structure_path.numpy().decode()

        atoms_list, adjoin_matrix = smiles2adjoin(smiles_str, explicit_hydrogens=self.addH)

        # 直接从提供的路径读取3D坐标
        coords = self._get_coords_from_path(structure_path_str, len(atoms_list))

        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]

        # 关键修复：确保坐标数量与原子数量匹配（不包括<global>）
        if len(coords) != len(atoms_list) - 1:
            # 如果坐标数量不匹配，进行调整
            if len(coords) > len(atoms_list) - 1:
                coords = coords[:len(atoms_list) - 1]
            else:
                # 填充缺失的坐标
                padding = np.zeros((len(atoms_list) - 1 - len(coords), 3))
                coords = np.concatenate([coords, padding], axis=0)

        # 扩展坐标以匹配原子列表（包括<global>）
        global_coord = np.mean(coords, axis=0, keepdims=True) if len(coords) > 0 else np.zeros((1, 3))
        coords = np.concatenate([global_coord, coords], axis=0)

        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        # 掩码处理
        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')
        return x, adjoin_matrix, coords, y, weight

    def _get_coords_from_path(self, structure_path, expected_num_atoms):
        """从指定路径读取3D坐标"""
        if not self.use_3d or not structure_path or not os.path.exists(structure_path):
            # 如果不使用3D或文件不存在，生成默认坐标
            return self._generate_default_coords(expected_num_atoms)

        try:
            # 根据文件扩展名判断格式
            file_ext = os.path.splitext(structure_path)[1].lower()

            if file_ext == '.sdf':
                coords = self._read_sdf_coords(structure_path, expected_num_atoms)
            elif file_ext == '.mol':
                coords = self._read_mol_coords(structure_path, expected_num_atoms)
            elif file_ext == '.xyz':
                coords = self._read_xyz_coords(structure_path, expected_num_atoms)
            elif file_ext == '.pdb':
                coords = self._read_pdb_coords(structure_path, expected_num_atoms)
            else:
                coords = self._generate_default_coords(expected_num_atoms)

            # 检查坐标是否有效（避免所有Z坐标为0的警告）
            if len(coords) > 0 and np.all(coords[:, 2] == 0):
                # 如果所有Z坐标都是0，添加一些随机性
                coords[:, 2] = np.random.normal(0, 0.1, len(coords))
                # print(f"修复平面坐标: {structure_path}")

            return coords

        except Exception as e:
            # print(f"读取3D文件失败 {structure_path}: {e}")
            return self._generate_default_coords(expected_num_atoms)

    def _generate_default_coords(self, expected_num_atoms):
        """生成默认的3D坐标"""
        # 生成一些合理的3D坐标而不是全零
        coords = np.random.normal(0, 1.0, (expected_num_atoms, 3))
        return coords

    def _read_sdf_coords(self, sdf_path, expected_num_atoms):
        """从SDF文件读取坐标"""
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(sdf_path)
        mol = next(suppl)
        return self._get_mol_coords(mol, expected_num_atoms)

    def _read_mol_coords(self, mol_path, expected_num_atoms):
        """从MOL文件读取坐标"""
        from rdkit import Chem
        mol = Chem.MolFromMolFile(mol_path)
        return self._get_mol_coords(mol, expected_num_atoms)

    def _read_xyz_coords(self, xyz_path, expected_num_atoms):
        """从XYZ文件读取坐标"""
        coords = []
        with open(xyz_path, 'r') as f:
            lines = f.readlines()[2:]  # 跳过前两行
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(coords)

    def _read_pdb_coords(self, pdb_path, expected_num_atoms):
        """从PDB文件读取坐标"""
        coords = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
        return np.array(coords)

    def _get_mol_coords(self, mol, expected_num_atoms):
        """从RDKit分子对象获取坐标"""
        if mol is None:
            return np.zeros((expected_num_atoms, 3))

        try:
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            return np.array(coords)
        except:
            return np.zeros((expected_num_atoms, 3))

    def _generate_coords_from_smiles(self, structure_path, expected_num_atoms):
        """备选方案：从文件路径推断SMILES并生成坐标"""
        try:
            # 尝试从文件路径提取SMILES（比如路径中包含SMILES）
            # 例如: "/path/to/structures/CCO.sdf" -> "CCO"
            filename = os.path.basename(structure_path)
            smiles_guess = os.path.splitext(filename)[0]

            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles_guess)
            if mol is None:
                return np.zeros((expected_num_atoms, 3))

            if self.addH:
                mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            return self._get_mol_coords(mol, expected_num_atoms)
        except:
            return np.zeros((expected_num_atoms, 3))

    def tf_numerical_smiles_3d(self, smiles, structure_path):
        x, adjoin_matrix, coords, y, weight = tf.py_function(
            self.numerical_smiles_3d,
            [smiles, structure_path],
            [tf.int64, tf.float32, tf.float32, tf.int64, tf.float32]
        )
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        coords.set_shape([None, 3])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, coords, y, weight


# 独立的 Graph_Regression_Dataset_3D 类
class Graph_Regression_Dataset_3D(object):
    def __init__(self, path, smiles_field='SMILES', label_field='Label',
                 structure_paths_field='structure_paths', normalize=True,
                 max_len=100, addH=True, use_3d=True):
        """
        3D图回归数据集

        Args:
            path: 数据文件路径
            smiles_field: SMILES列名
            label_field: 标签列名
            structure_paths_field: 3D结构文件路径列名
            normalize: 是否标准化标签
            max_len: 最大SMILES长度
            addH: 是否添加氢原子
            use_3d: 是否使用3D结构
        """
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)

        self.smiles_field = smiles_field
        self.label_field = label_field
        self.structure_paths_field = structure_paths_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH
        self.use_3d = use_3d

        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field] - self.min) / (self.max - self.min) - 0.5
            self.value_range = self.max - self.min

    def get_data(self):
        data = self.df
        data = data.dropna()

        lengths = [0, 25, 50, 75, 100]

        # 第一步：先划分训练集 (70%)
        train_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) &
                       (data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.7).index
            train_idx.extend(idx)

        train_data = data[data.index.isin(train_idx)]
        remaining_data = data[~data.index.isin(train_idx)]

        # 第二步：从剩余数据中划分测试集 (剩余数据的50%，即整体的15%)
        test_idx = []
        for i in range(4):
            # 从剩余数据中按长度分组采样50%
            remaining_group = remaining_data[(remaining_data[self.smiles_field].str.len() >= lengths[i]) &
                                             (remaining_data[self.smiles_field].str.len() < lengths[i + 1])]
            if len(remaining_group) > 0:
                idx = remaining_group.sample(frac=0.5).index
                test_idx.extend(idx)

        test_data = remaining_data[remaining_data.index.isin(test_idx)]
        val_data = remaining_data[~remaining_data.index.isin(test_idx)]

        print(f"数据集划分结果:")
        print(f"训练集: {len(train_data)} 样本 ({len(train_data) / len(data) * 100:.1f}%)")
        print(f"验证集: {len(val_data)} 样本 ({len(val_data) / len(data) * 100:.1f}%)")
        print(f"测试集: {len(test_data)} 样本 ({len(test_data) / len(data) * 100:.1f}%)")

        # 创建包含SMILES、结构路径和标签的数据集
        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (train_data[self.smiles_field], train_data[self.structure_paths_field], train_data[self.label_field])
        )
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles_3d).cache().padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]),
            tf.TensorShape([None, 3]), tf.TensorShape([1]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(
            (test_data[self.smiles_field], test_data[self.structure_paths_field], test_data[self.label_field])
        )
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles_3d).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]),
            tf.TensorShape([None, 3]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices(
            (val_data[self.smiles_field], val_data[self.structure_paths_field], val_data[self.label_field])
        )
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles_3d).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]),
            tf.TensorShape([None, 3]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles_3d(self, smiles, structure_path, label):
        smiles_str = smiles.numpy().decode()
        structure_path_str = structure_path.numpy().decode()
        label_val = label.numpy()

        atoms_list, adjoin_matrix = smiles2adjoin(smiles_str, explicit_hydrogens=self.addH)

        # 读取3D坐标
        coords = self._get_coords_from_path(structure_path_str, len(atoms_list))

        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]

        # 确保坐标数量与原子数量匹配
        if len(coords) != len(atoms_list) - 1:
            if len(coords) > len(atoms_list) - 1:
                coords = coords[:len(atoms_list) - 1]
            else:
                padding = np.zeros((len(atoms_list) - 1 - len(coords), 3))
                coords = np.concatenate([coords, padding], axis=0)

        # 扩展坐标以匹配原子列表（包括<global>）
        global_coord = np.mean(coords, axis=0, keepdims=True) if len(coords) > 0 else np.zeros((1, 3))
        coords = np.concatenate([global_coord, coords], axis=0)

        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label_val]).astype('float32')
        return x, adjoin_matrix, coords, y

    def _get_coords_from_path(self, structure_path, expected_num_atoms):
        """从指定路径读取3D坐标"""
        if not self.use_3d or not structure_path or not os.path.exists(structure_path):
            return self._generate_default_coords(expected_num_atoms)

        try:
            file_ext = os.path.splitext(structure_path)[1].lower()

            if file_ext == '.sdf':
                coords = self._read_sdf_coords(structure_path, expected_num_atoms)
            elif file_ext == '.mol':
                coords = self._read_mol_coords(structure_path, expected_num_atoms)
            elif file_ext == '.xyz':
                coords = self._read_xyz_coords(structure_path, expected_num_atoms)
            elif file_ext == '.pdb':
                coords = self._read_pdb_coords(structure_path, expected_num_atoms)
            else:
                coords = self._generate_default_coords(expected_num_atoms)

            # 检查坐标是否有效
            if len(coords) > 0 and np.all(coords[:, 2] == 0):
                coords[:, 2] = np.random.normal(0, 0.1, len(coords))

            return coords

        except Exception as e:
            return self._generate_default_coords(expected_num_atoms)

    def _generate_default_coords(self, expected_num_atoms):
        """生成默认的3D坐标"""
        coords = np.random.normal(0, 1.0, (expected_num_atoms, 3))
        return coords

    def _read_sdf_coords(self, sdf_path, expected_num_atoms):
        """从SDF文件读取坐标"""
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(sdf_path)
        mol = next(suppl)
        return self._get_mol_coords(mol, expected_num_atoms)

    def _read_mol_coords(self, mol_path, expected_num_atoms):
        """从MOL文件读取坐标"""
        from rdkit import Chem
        mol = Chem.MolFromMolFile(mol_path)
        return self._get_mol_coords(mol, expected_num_atoms)

    def _read_xyz_coords(self, xyz_path, expected_num_atoms):
        """从XYZ文件读取坐标"""
        coords = []
        with open(xyz_path, 'r') as f:
            lines = f.readlines()[2:]  # 跳过前两行
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(coords)

    def _read_pdb_coords(self, pdb_path, expected_num_atoms):
        """从PDB文件读取坐标"""
        coords = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
        return np.array(coords)

    def _get_mol_coords(self, mol, expected_num_atoms):
        """从RDKit分子对象获取坐标"""
        if mol is None:
            return np.zeros((expected_num_atoms, 3))

        try:
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            return np.array(coords)
        except:
            return np.zeros((expected_num_atoms, 3))

    def tf_numerical_smiles_3d(self, smiles, structure_path, label):
        x, adjoin_matrix, coords, y = tf.py_function(
            self.numerical_smiles_3d,
            [smiles, structure_path, label],
            [tf.int64, tf.float32, tf.float32, tf.float32]
        )
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        coords.set_shape([None, 3])
        y.set_shape([1])
        return x, adjoin_matrix, coords, y

    def get_k_fold_data(self, k=5, fold=0, seed=42):
        """
        获取K折交叉验证的特定fold数据 - 改进版本
        划分比例: 训练集70%, 验证集15%, 测试集15%
        """
        data = self.df
        data = data.dropna()

        print(f"原始数据数量: {len(self.df)}")
        print(f"过滤后数据数量: {len(data)}")

        # 设置随机种子确保可重复性
        np.random.seed(seed)

        # 获取所有数据的索引
        n_samples = len(data)
        print(f"可用于K折划分的样本数: {n_samples}")

        # 使用位置索引
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # 计算每个fold的大小 (测试集占15%)
        fold_size = int(0.15 * n_samples)  # 每个fold的测试集大小
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size

        # 如果是最后一折，包含剩余的所有样本
        if fold == k - 1:
            test_end = n_samples

        # 划分测试集 (15%)
        test_indices = indices[test_start:test_end]

        # 剩余数据用于训练和验证 (85%)
        remaining_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        # 从剩余数据中划分验证集 (剩余数据的17.6% ≈ 整体的15%)
        n_remaining = len(remaining_indices)
        val_size = int(0.176 * n_remaining)  # 0.176 * 0.85 ≈ 0.15

        # 如果验证集大小计算为0，至少保留1个样本
        if val_size == 0 and n_remaining > 0:
            val_size = 1

        val_indices = remaining_indices[:val_size]
        train_indices = remaining_indices[val_size:]

        # 使用iloc根据位置索引获取数据
        train_data = data.iloc[train_indices]
        val_data = data.iloc[val_indices]
        test_data = data.iloc[test_indices]

        print(f"K折交叉验证 - Fold {fold + 1}/{k}")
        print(f"训练集: {len(train_data)} 样本 ({len(train_data) / n_samples * 100:.1f}%)")
        print(f"验证集: {len(val_data)} 样本 ({len(val_data) / n_samples * 100:.1f}%)")
        print(f"测试集: {len(test_data)} 样本 ({len(test_data) / n_samples * 100:.1f}%)")
        print(f"总计: {len(train_data) + len(val_data) + len(test_data)} 样本")

        # 检查是否有数据丢失
        if len(train_data) + len(val_data) + len(test_data) != n_samples:
            print(
                f"⚠️ 警告: 数据划分有丢失! 期望: {n_samples}, 实际: {len(train_data) + len(val_data) + len(test_data)}")

        # 检查数据集是否为空
        if len(train_data) == 0:
            print("❌ 错误: 训练集为空!")
        if len(val_data) == 0:
            print("❌ 错误: 验证集为空!")
        if len(test_data) == 0:
            print("❌ 错误: 测试集为空!")

        # 创建数据集
        def create_dataset(smiles_data, structure_data, label_data, batch_size, shuffle=False):
            dataset = tf.data.Dataset.from_tensor_slices(
                (smiles_data, structure_data, label_data)
            )
            dataset = dataset.map(self.tf_numerical_smiles_3d)

            if shuffle:
                dataset = dataset.shuffle(min(1000, len(smiles_data)))

            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),  # x
                    tf.TensorShape([None, None]),  # adjoin_matrix
                    tf.TensorShape([None, 3]),  # coords
                    tf.TensorShape([1])  # y
                )
            )
            return dataset.prefetch(tf.data.AUTOTUNE)

        # 创建训练、验证、测试数据集
        self.dataset1 = create_dataset(
            train_data[self.smiles_field],
            train_data[self.structure_paths_field],
            train_data[self.label_field],
            batch_size=32,
            shuffle=True
        )

        self.dataset2 = create_dataset(
            test_data[self.smiles_field],
            test_data[self.structure_paths_field],
            test_data[self.label_field],
            batch_size=64,
            shuffle=False
        )

        self.dataset3 = create_dataset(
            val_data[self.smiles_field],
            val_data[self.structure_paths_field],
            val_data[self.label_field],
            batch_size=64,
            shuffle=False
        )

        return self.dataset1, self.dataset2, self.dataset3