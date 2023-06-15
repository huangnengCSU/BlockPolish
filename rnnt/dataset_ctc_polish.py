import codecs
import copy
import numpy as np
import torch.utils.data as Data
import os
import math
from multiprocessing import Queue,Process,Manager
import torch

# from utils import AttrDict, init_logger, count_parameters, save_model, computer_cer
# import yaml
# import torch
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

base2int = {'A': 1, 'A+': 2, 'C': 3, 'C+': 4, 'G': 5, 'G+': 6, 'T': 7, 'T+': 8, 'N': 9, 'N+': 10}  # padding 0
int2base = {1: 'A', 2: 'a', 3: 'C', 4: 'c', 5: 'G', 6: 'g', 7: 'T', 8: 't', 9: 'N', 10: 'n'}  # padding 0


class FeatureArray():
    refName = ""
    intervalStart = -1
    intervalEnd = -1
    array = None

    def __init__(self, name, s, e, array):
        self.refName = name
        self.intervalStart = s
        self.intervalEnd = e
        self.array = array


def generate_flipflop_sequence(value, length):
    out = []
    for i in range(length):
        if i % 2 == 0:
            out.append(value)
        else:
            out.append(value + 1)
    return out


def rle_label(seq, max_rle):
    base_list, rle_list = [], []
    rle = 1
    for i in range(len(seq)):
        if i > 0:
            prev_base = seq[i - 1]
            cur_base = seq[i]
            if prev_base == cur_base:
                rle += 1
            else:
                base_list.append(prev_base)
                if rle >= max_rle:
                    rle = max_rle
                # rle_list.append(rle_encoding[prev_base][rle - 1])
                rle_list.extend(generate_flipflop_sequence(prev_base, rle))
                rle = 1
        if i == len(seq) - 1:
            base_list.append(seq[i])
            if rle >= max_rle:
                rle = max_rle
            rle_list.extend(generate_flipflop_sequence(seq[i], rle))
    return np.array(base_list), np.array(rle_list)


def load_train_data(datafile, max_rle):
    regions, feats, targets, rle_bases, rles = [], [], [], [], []
    feat_max_length = 0
    target_max_length = 0
    rle_base_max_length = 0
    rle_max_length = 0
    with open(datafile) as fin:
        for line in fin:
            if line.startswith(">"):
                header = line.rstrip().replace('>', '')
                refName, startPos, endPos, label = header.split('\t')
                label = encode(label)
                label_shape = label.shape
                rle_base, rle = rle_label(label, max_rle)
            else:
                array = np.array([float(v) for v in line.split(',')[:-1]]).reshape((-1, 7))
                array_shape = array.shape
                if array_shape[0] > feat_max_length:
                    feat_max_length = array_shape[0]
                if rle_base.shape[0] > array_shape[0] or rle.shape[0] > array_shape[0]:
                    # print(rle_base.shape, rle.shape, array_shape)
                    # filter target length greater than input length
                    continue
                if label_shape[0] > 100 or array_shape[0] > 100:
                    # filter the extra long feature or target
                    continue
                if label_shape[0] > target_max_length:
                    target_max_length = label_shape[0]
                if rle_base.shape[0] > rle_base_max_length:
                    rle_base_max_length = rle_base.shape[0]
                if rle.shape[0] > rle_max_length:
                    rle_max_length = rle.shape[0]
                feats.append(array)
                targets.append(label)
                rle_bases.append(rle_base)
                rles.append(rle)
                regions.append(refName + '\t' + startPos + '\t' + endPos)
    return regions, feats, feat_max_length, targets, target_max_length, rle_bases, rles, rle_base_max_length, rle_max_length


def load_feature_data(datafile):
    regions, feats = [], []
    feat_max_length = 0
    with open(datafile) as fin:
        for line in fin:
            if line.startswith(">"):
                header = line.rstrip().replace('>', '')
                refName, startPos, endPos = header.split('\t')
            else:
                array = np.array([float(v) for v in line.split(',')[:-1]]).reshape((-1, 7))
                array_shape = array.shape
                if array_shape[0] > feat_max_length:
                    feat_max_length = array_shape[0]

                #  超长的简单区域或者复杂区域，分割成小的区域
                if array_shape[0] > 100:
                    split_array_list = []
                    split_region_list = []
                    split_size = math.ceil(array_shape[0] / 100)
                    for i in range(split_size):
                        if i != split_size - 1:
                            split_array_list.append(array[i * 100:(i + 1) * 100])
                        else:
                            split_array_list.append(array[i * 100:array_shape[0]])
                        split_region_list.append(refName + '\t' + startPos + '\t' + endPos + "\t" + str(i))
                    feats.extend(split_array_list)
                    regions.extend(split_region_list)
                    continue
                feats.append(array)
                regions.append(refName + '\t' + startPos + '\t' + endPos + "\t" + str(0))  # contig+起点+终点+重复编号
    return regions, feats, feat_max_length


def encode(seq):
    array = []
    for base in seq:
        array.append(base2int[base])
    return np.array(array)


def pad(inputs, max_length):
    dim = len(inputs.shape)
    if dim == 1:
        pad_zeros_mat = np.zeros([max_length - inputs.shape[0]], dtype=np.int32)
        padded_inputs = np.concatenate([inputs, pad_zeros_mat])
    elif dim == 2:
        feature_dim = inputs.shape[1]
        pad_zeros_mat = np.zeros([max_length - inputs.shape[0], feature_dim])
        padded_inputs = np.row_stack([inputs, pad_zeros_mat])
    else:
        raise AssertionError(
            'Features in inputs list must be one or two dimension matrix! ')
    return padded_inputs


class PolishTrainDataset(Data.Dataset):
    def __init__(self, datafile, max_rle):
        super(PolishTrainDataset, self).__init__()
        self.train_data_file = datafile
        self.regions, self.train_feats, self.feat_max_length, self.train_labels, self.label_max_length, self.rle_bases, self.rles, self.rle_base_max_length, self.rle_max_length = load_train_data(
            datafile, max_rle)  # [array(L,7),array(L,7),...]
        self.lengths = len(self.train_feats)

    def __getitem__(self, index):
        region = self.regions[index]
        feature = self.train_feats[index]
        target = self.train_labels[index]
        rle_base = self.rle_bases[index]
        rle = self.rles[index]
        input_length = np.array(feature.shape[0]).astype(np.int)
        target_length = np.array(target.shape[0]).astype(np.int)
        rle_base_length = np.array(rle_base.shape[0]).astype(np.int)
        rle_length = np.array(rle.shape[0]).astype(np.int)
        feature = pad(feature, self.feat_max_length).astype(np.float32)  # [array(L,7)]
        target = pad(target, self.label_max_length).astype(np.int)  # [array(L)]
        rle_base = pad(rle_base, self.rle_base_max_length).astype(np.int)
        rle = pad(rle, self.rle_max_length).astype(np.int)
        return region, feature, input_length, target, target_length, rle_base, rle_base_length, rle, rle_length

    def __len__(self):
        return self.lengths

def Prod(datafile,batch_size,thread_num,prod_queue):
    """生产者线程"""
    tmp = []
    idx = 0
    with open(datafile) as fin:
        for line in fin:
            if line.startswith(">"):
                header = line.rstrip()
            else:
                feat = line.rstrip()
                tmp.append((idx,header,feat))
                if len(tmp)==batch_size:
                    prod_queue.put(tmp)
                    idx+=1
                    tmp = []
    return 0

def Cust(prod_queue,cust_queue):
    while True:
        batch_data = prod_queue.get()
        if batch_data is None:
            break
        """padding"""
        regions, feats = [],[]
        feat_max_length = 0
        for (idx,header,feat) in batch_data:
            header = header.rstrip().replace('>', '')
            refName, startPos, endPos = header.split('\t')

            array = np.array([float(v) for v in feat.split(',')[:-1]]).reshape((-1, 7))
            array_shape = array.shape
            #  超长的简单区域或者复杂区域，分割成小的区域
            if array_shape[0] > 100:
                split_array_list = []
                split_region_list = []
                split_size = math.ceil(array_shape[0] / 100)
                for i in range(split_size):
                    if i != split_size - 1:
                        sarray = array[i * 100:(i + 1) * 100]   # [100,7]
                        if sarray.shape[0] > feat_max_length:
                            feat_max_length = sarray.shape[0]   #   100
                        split_array_list.append(sarray)
                    else:
                        sarray = array[i * 100:array_shape[0]]  # [n,7]
                        if sarray.shape[0] > feat_max_length:
                            feat_max_length = sarray.shape[0]
                        split_array_list.append(sarray)
                    split_region_list.append(refName + '\t' + startPos + '\t' + endPos + "\t" + str(i))
                feats.extend(split_array_list)
                regions.extend(split_region_list)
                continue
            if array_shape[0] > feat_max_length:
                feat_max_length = array_shape[0]
            feats.append(array)
            regions.append(refName + '\t' + startPos + '\t' + endPos + "\t" + str(0))  # contig+起点+终点+重复编号
        feature_lengths = np.array([v.shape[0] for v in feats]).astype(np.int)
        feature = np.array([pad(v, feat_max_length).astype(np.float32) for v in feats])
        cust_queue.put((idx,feature,regions,feature_lengths))
    return 0
        

def load_feature_generate_data(datafile,batch_size, thread_num):
    pthread_list = []
    qMar = Manager()
    prod_queue = qMar.Queue(maxsize=100)
    cust_queue = qMar.Queue()

    ##建立1个生产者线程
    thread_pro = Process(target=Prod,args=[datafile,batch_size,thread_num,prod_queue])
    thread_pro.start()

    ##建立多个个消费者线程
    for i in range(thread_num):
        thread_cus = Process(target=Cust, args=[prod_queue,cust_queue])
        thread_cus.start()
        pthread_list.append(thread_cus)
    
    thread_pro.join()
    
    for i in range(thread_num):
        prod_queue.put(None)

    for t in pthread_list:
        t.join()
    
    print("threading done.")
    return cust_queue


class PolishGenerateDataset(Data.Dataset):
    def __init__(self, datafile,batch_size,threads):
        super(PolishGenerateDataset, self).__init__()
        self.train_data_file = datafile
        data_queue = load_feature_generate_data(datafile,batch_size,threads)
        # self.regions, self.train_feats, self.feat_max_length = load_feature_data(datafile)  # [array(L,7),array(L,7),...]
        self.lengths = data_queue.qsize()
        batch_data_dict = {}
        while not data_queue.empty():
            (idx,features,regions,input_lengths) = data_queue.get()
            batch_data_dict[idx] = (features,regions,input_lengths)
        self.batch_data_dict = batch_data_dict

    def __getitem__(self, index):
        batch_data = self.batch_data_dict[index]
        regions = batch_data[1]
        features = batch_data[0]
        input_lengths = batch_data[2]
        return regions, features, input_lengths

    def __len__(self):
        return self.lengths

class PolishPredictDataset(Data.Dataset):
    def __init__(self, datafile):
        super(PolishPredictDataset, self).__init__()
        self.train_data_file = datafile
        # self.regions, self.train_feats, self.feat_max_length = load_feature_data(datafile)
        fopen = open(datafile,'r')
        length = 0
        for line in fopen:
            if line.startswith(">"):
                length += 1
        self.lengths = math.ceil(length/1000)
        fopen.close()
        self.fopen = open(datafile,'r')
    
    def __getitem__(self, index):
        regions, feats = [],[]
        feat_max_length = 0
        for i in range(1000):
            header = self.fopen.readline()
            if not header:
                break
            feat = self.fopen.readline()
            if not feat:
                break
            header = header.rstrip().replace('>', '')
            refName, startPos, endPos = header.split('\t')
            array = np.array([float(v) for v in feat.split(',')[:-1]]).reshape((-1, 7)) # [L,7]
            array_shape = array.shape   # [L,7]
            if array_shape[0] > 100:
                split_array_list = []
                split_region_list = []
                split_size = math.ceil(array_shape[0] / 100)
                for i in range(split_size):
                    if i != split_size - 1:
                        sarray = array[i * 100:(i + 1) * 100]
                        if sarray.shape[0] > feat_max_length:
                            feat_max_length = sarray.shape[0]
                        split_array_list.append(sarray)
                    else:
                        sarray = array[i * 100:array_shape[0]]
                        if sarray.shape[0] > feat_max_length:
                            feat_max_length = sarray.shape[0]
                        split_array_list.append(sarray)
                    split_region_list.append(refName + '\t' + startPos + '\t' + endPos + "\t" + str(i))
                feats.extend(split_array_list)
                regions.extend(split_region_list)
            else:
                if array_shape[0] > feat_max_length:
                    feat_max_length = array_shape[0]
                feats.append(array)
                regions.append(refName + '\t' + startPos + '\t' + endPos + "\t" + str(0))
        feature_lengths = np.array([v.shape[0] for v in feats]).astype(np.int)
        feature = np.array([pad(v, feat_max_length).astype(np.float32) for v in feats])
        return regions, feature, feature_lengths
    
    def __len__(self):
        return self.lengths

        

# if __name__ == '__main__':
#     configfile = open("../config/aishell.yaml")
#
#     config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
#     train_dataset = PolishTrainDataset(config.data)
#     training_data = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
#     print(train_dataset.feat_max_length, train_dataset.label_max_length)
#     for setp, (regions, feats, inputs_length, targets, targets_length) in enumerate(training_data):
#         feats = torch.FloatTensor(feats)
#         targets = torch.LongTensor(targets)
#         inputs_length = torch.LongTensor(inputs_length)
#         targets_length = torch.LongTensor(targets_length)
#         pack = pack_padded_sequence(feats, inputs_length, batch_first=True, enforce_sorted=False)  # [N, L, C]

if __name__=='__main__':
    # generate_dataset = PolishGenerateDataset("/home/user/experiment/BlockPolish/computing_time/BlockPolish/racon/chr22-complex_features.txt",512,40)
    # generateg_data = torch.utils.data.DataLoader(generate_dataset, batch_size=1, shuffle=False, num_workers=4)
    # for setp, (regions, feats, inputs_length) in enumerate(generateg_data):
    #     print(feats.shape,inputs_length.shape)
    #     # print(inputs_length[0])
    #     print(regions)
    #     feats = torch.FloatTensor(feats)
    #     inputs_length = torch.LongTensor(inputs_length)
    #     print(feats.size())

    generate_dataset = PolishPredictDataset("~/projects/BlockPolish/hg002/trivial.txt")
    generateg_data = torch.utils.data.DataLoader(generate_dataset, batch_size=1, shuffle=False, num_workers=0)
    max_len = 0
    for setp, (regions, feats, inputs_length) in enumerate(generateg_data):
        regions = regions
        feats = feats[0]
        inputs_length = inputs_length[0]
        # print(regions,feats,inputs_length)
        # print(feats.shape,inputs_length.shape)
        # print(inputs_length[0])
        # print(regions)
        feats = torch.FloatTensor(feats)
        inputs_length = torch.LongTensor(inputs_length)
        if len(regions) > max_len:
            max_len = len(regions)
        print(len(regions),'\t', feats.size(), '\t', inputs_length.size())
    print(max_len)
