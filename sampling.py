import math
import os
import numpy as np
import random

import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import pdb


def under_sampling(dataset):
    p=0.1
    max=1000
    ind1=[]
    ind =[]
    classes = np.unique(dataset.labels)
    class_num=len(classes)
    min_class = round(max * p)

    for i in range(class_num):
        index = np.where(dataset.labels == i)[0]
        ind1 = index[0:min_class]
        ind.extend(ind1)
    ind.sort()
    dataset.data= dataset.data[ind]
    dataset.label = dataset.labels[ind]

    return dataset

def under_sampling_mnist(dataset):
    img = dataset.imgs
    classes = dataset.classes

    max=1000
    p = 0.1
    min_class = round(max*p)
    ind1 = []

    labels = []
    for item in range(len(img)):
        labels.append(img[item][1])

    for i in range(len(classes)):
        index = np.where(labels == i)[0]
        ind1 = index[0:min_class]
        ind.extend(ind1)
    ind.sort()
    dataset.imgs = dataset.imgs[ind]
    return dataset


"""
class-aware sampling code，from decoupling
"""

class RandomCycleIter:
    def __init__(self, data,test_mode =False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


#逐个采样,num_samples=cls
def class_aware_sample_generator(class_iter, data_iter_list, n, num_samples_cls):
    """
    :param class_iter:
    :param data_iter_list:
    :param n: number of all examples
    :param num_samples_cls: sampled numbers in each class
    :return:
    """
    i = 0
    j = 0
    while i < n:
        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(class_iter)]] * num_samples_cls))

            yield temp_tuple[j]

        else:
            yield temp_tuple[j]

        i += 1
        j += 1



class ClassAwareSampler(Sampler):

    def __init__(self, source_label, batchsize):
        super().__init__(source_label)
        # datasets_dict = {'SVHNRGB': 'SVHN', 'MNISTRGB': 'MNIST', 'USPS': 'USPS', 'MNIST': 'MNIST'}
        train_label =  source_label
        classes = np.unique(train_label)
        self.num_classes = len(classes)
        self.class_data_list = [list() for _ in range(self.num_classes)]
        self.class_iter = RandomCycleIter(range(self.num_classes))
        for i, label in enumerate(train_label):
            self.class_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in self.class_data_list]
        self.num_samples = max([len(x) for x in self.class_data_list]) * self.num_classes
        self.num_samples_cls = round(batchsize/self.num_classes)
        # self.num_samples_cls = math.ceil(batchsize / self.num_classes)
        print (self.num_samples)


    def __iter__(self):
        # return class_aware_sample_generator(self.class_iter, self.data_iter_list,
        #                                      self.num_samples , self.num_samples_cls)
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples


# class ClassAwareSampler(Sampler):
#
#     def __init__(self, data_source, total_samples, beta=0.999, shuffle=True):
#         labels = data_source.dataset[data_source.label_key]
#
#         #labels = data_source.targets
#
#         num_classes = len(np.unique(labels))
#         label_to_count = [0] * num_classes
#
#         for idx, label in enumerate(labels):
#             label_to_count[label] += 1
#
#         if beta < 1:
#             effective_num = 1.0 - np.power(beta, label_to_count)
#             per_cls_weights = (1.0 - beta) / np.array(effective_num)
#         else:
#             per_cls_weights = 1.0 / np.array(label_to_count)
#
#         weights = torch.DoubleTensor([per_cls_weights[label] for label in labels])
#
#         # total train epochs
#         num_epochs = int(total_samples / len(labels)) + 1
#         total_inds = []
#         for epoch in range(num_epochs):
#             inds_list = torch.multinomial(weights, len(labels), replacement=True).tolist()
#             if shuffle:
#                 random.shuffle(inds_list)
#             total_inds.extend(inds_list)
#         total_inds = total_inds[:total_samples]
#
#         self.per_cls_prob = per_cls_weights / np.sum(per_cls_weights)
#
#         self._indices = total_inds
#
#     def __iter__(self):
#         return iter(self._indices)
#
#     def __len__(self):
#         return len(self._indices)



class UniformResamplingSampler(Sampler):

    def __init__(self, data_source, dataname, batchsize):
        super().__init__(data_source)
        self.data_source = data_source
        train_label = data_source.labels if dataname == 'SVHN' else data_source.targets
        self.classes = np.unique(train_label)
        num_classes = len(self.classes)
        self.class_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(train_label):
            self.class_data_list[label].append(i)

        self.num_samples = max([len(x) for x in self.class_data_list]) * num_classes
        self.num_samples_cls = round(batchsize / num_classes)
        self.sample_iter = []

    def __iter__(self):
        for i in self.classes:
            # class_iter = random.choice(self.class_data_list[i], self.num_samples_cls)
            class_iter = random.choice(self.class_data_list[i])
            self.sample_iter.append(class_iter)
        return iter(self.sample_iter)

    def __len__(self):
        return self.num_samples