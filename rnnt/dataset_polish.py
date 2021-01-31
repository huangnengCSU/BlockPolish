import codecs
import copy
import numpy as np
import torch.utils.data as Data
import os

# from utils import AttrDict, init_logger, count_parameters, save_model, computer_cer
# import yaml
# import torch
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

base2int = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}  # padding 0


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


def rle_label(seq):
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
                rle_list.append(rle)
                rle = 1
        if i == len(seq) - 1:
            base_list.append(seq[i])
            rle_list.append(rle)
    return np.array(base_list), np.array(rle_list)


def load_train_data(datafile):
    regions, feats, targets, rle_bases, rles = [], [], [], [], []
    feat_max_length = 0
    target_max_length = 0
    rle_max_length = 0
    with open(datafile) as fin:
        for line in fin:
            if line.startswith(">"):
                header = line.rstrip().replace('>', '')
                refName, startPos, endPos, label = header.split('\t')
                label = encode(label)
                label_shape = label.shape
                if label_shape[0] > target_max_length:
                    target_max_length = label_shape[0]
                rle_base, rle = rle_label(label)
                if rle_base.shape[0] > rle_max_length:
                    rle_max_length = rle_base.shape[0]
                targets.append(label)
                rle_bases.append(rle_base)
                rles.append(rle)
                regions.append(refName + '\t' + startPos + '\t' + endPos)
            else:
                array = np.array([float(v) for v in line.split(',')[:-1]]).reshape((-1, 7))
                array_shape = array.shape
                if array_shape[0] > feat_max_length:
                    feat_max_length = array_shape[0]
                feats.append(array)
    return regions, feats, feat_max_length, targets, target_max_length, rle_bases, rles, rle_max_length


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
    def __init__(self, datafile):
        super(PolishTrainDataset, self).__init__()
        self.train_data_file = datafile
        self.regions, self.train_feats, self.feat_max_length, self.train_labels, self.label_max_length, self.rle_bases, self.rles, self.rle_max_length = load_train_data(
            datafile)  # [array(L,7),array(L,7),...]
        self.lengths = len(self.train_feats)

    def __getitem__(self, index):
        region = self.regions[index]
        feature = self.train_feats[index]
        target = self.train_labels[index]
        rle_base = self.rle_bases[index]
        rle = self.rles[index]
        input_length = np.array(feature.shape[0]).astype(np.int)
        target_length = np.array(target.shape[0]).astype(np.int)
        rle_length = np.array(rle_base.shape[0]).astype(np.int)
        feature = pad(feature, self.feat_max_length).astype(np.float32)  # [array(L,7)]
        target = pad(target, self.label_max_length).astype(np.int)  # [array(L)]
        rle_base = pad(rle_base, self.rle_max_length).astype(np.int)
        rle = pad(rle,self.rle_max_length).astype(np.int)
        return region, feature, input_length, target, target_length, rle_base, rle, rle_length

    def __len__(self):
        return self.lengths


class PolishTestDataset(Data.Dataset):
    def __init__(self, datafile):
        super(PolishTestDataset, self).__init__()
        self.train_data_file = datafile
        self.regions, self.train_feats, self.feat_max_length, self.train_labels, self.label_max_length = load_train_data(
            datafile)  # [array(L,7),array(L,7),...]
        self.lengths = len(self.train_feats)

    def __getitem__(self, index):
        region = self.regions[index]
        feature = self.train_feats[index]
        target = self.train_labels[index]
        input_length = np.array(feature.shape[0]).astype(int).astype(np.int)
        target_length = np.array(target.shape[0]).astype(int).astype(np.int)
        feature = pad(feature, self.feat_max_length).astype(np.float32)  # [array(L,7)]
        target = pad(target, self.label_max_length).astype(np.int)  # [array(L)]
        return region, feature, input_length, target, target_length

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
