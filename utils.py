import numpy as np
from torch.utils import data
# import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler
import torch
import os

# from sklearn.manifold import TSNE
# asc_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
#           '#00ff7f', '#9400d3', '#3b3b3b', '#0000ee', '#bcd2ee']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PairLoader(data.Dataset):
    def __init__(self, X, y):
        super(PairLoader, self).__init__()
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Xtensor = torch.tensor(self.data[idx], dtype=torch.float).to(device)
        ytensor = torch.tensor(self.labels[idx], dtype=torch.long).to(device)

        return [Xtensor, ytensor]


class PairLoader_large(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_IDs, labels):
        """Initialization"""
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        Xtensor = torch.tensor(np.load('data/' + ID + '.npy'), dtype=torch.float).to(device)
        ytensor = torch.tensor(self.labels[ID], dtype=torch.long).to(device)
        # X = torch.from_numpy(np.load('data/' + ID + '.npy'))
        # y = self.labels[ID]

        return Xtensor, ytensor


class BalanceBatchSampler_large(BatchSampler):
    """
    batch sampler, randomly select n_classes, and n_samples each class
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        labels_array = np.array(list(self.labels.values()))
        self.labels_set = list(set(labels_array))
        self.labels_to_indices = {label: np.where(labels_array == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.labels_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.dataset = dataset

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = self.labels_set[int(self.count / 4):int(self.count / 4 + self.n_classes)]
            # classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.labels_to_indices[class_][self.used_label_indices_count[class_]:
                                                              self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.labels_to_indices[class_]):
                    np.random.shuffle(self.labels_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size
# def preprocess(x, y, nb_classes=10, max_value=255):
#     x = x.astype('float32') / max_value
#     y = to_categorical(y, nb_classes)
#     return x, y
#
#
# def to_categorical(labels, nb_classes=None):
#     labels = np.array(labels, dtype=np.int32)
#     if not nb_classes:
#         nb_classes = np.max(labels) + 1
#     categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
#     categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
#     return categorical


class BalanceBatchSampler(BatchSampler):
    """
    batch sampler, randomly select n_classes, and n_samples each class
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels))
        self.labels_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.labels_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.dataset = dataset

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = self.labels_set[int(self.count / 4):int(self.count / 4 + self.n_classes)]
            # classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.labels_to_indices[class_][self.used_label_indices_count[class_]:
                                                              self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.labels_to_indices[class_]):
                    np.random.shuffle(self.labels_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size


class Reporter(object):
    def __init__(self, ckpt_root, exp, ckpt_file=None):
        self.ckpt_root = ckpt_root
        self.exp_path = os.path.join(self.ckpt_root, exp)
        self.run_list = os.listdir(self.exp_path)
        self.selected_ckpt = None
        self.selected_epoch = None
        self.selected_log = None
        self.selected_run = None

    def select_best(self, run=""):

        """
        set self.selected_run, self.selected_ckpt, self.selected_epoch
        :param run:
        :return:
        """

        matched = []
        for fname in self.run_list:
            if fname.startswith(run) and fname.endswith('tar'):
                matched.append(fname)

        loss = []
        import re
        for s in matched:
            acc_str = re.search('loss_(.*)\.tar', s).group(1)
            loss.append(float(acc_str))

        loss = np.array(loss)
        best_idx = np.argmin(loss)
        best_fname = matched[best_idx]

        self.selected_run = best_fname.split(',')[0]
        self.selected_epoch = int(re.search('Epoch_(.*),loss', best_fname).group(1))

        ckpt_file = os.path.join(self.exp_path, best_fname)

        self.selected_ckpt = ckpt_file

        return self

# def plot_embeddings(embeddings, targets, cls_num=10, xlim=None, ylim=None, title=None):
#     plt.figure(figsize=(10, 10))
#     # TODO init ?
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     result = tsne.fit_transform(embeddings)
#
#     for i in range(cls_num):
#         inds = np.where(targets == i)[0]
#
#         plt.scatter(result[inds, 0], result[inds, 1], c=colors[i])
#
#     if xlim:
#         plt.xlim(xlim[0], xlim[1])
#     if ylim:
#         plt.ylim(ylim[0], ylim[1])
#     if cls_num == 10:
#         plt.legend(asc_classes[:10])
#     else:
#         plt.legend(asc_classes)
#     if title:
#         plt.title(title)
#     plt.show()
