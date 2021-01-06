import numpy as np
from torch.utils import data
from torch.utils.data.sampler import BatchSampler
import torch
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PairLoader(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_IDs, labels, data_source='data_resized'):
        """Initialization"""
        self.labels = labels
        self.list_IDs = list_IDs
        self.data_source = data_source

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        Xtensor = torch.tensor(np.load(ROOT_DIR + '/' + self.data_source + '/' + ID + '.npy'), dtype=torch.float).to(device)
        ytensor = torch.tensor(self.labels[ID], dtype=torch.long).to(device)
        # X = torch.from_numpy(np.load('data/' + ID + '.npy'))
        # y = self.labels[ID]

        return Xtensor, ytensor


class BalanceBatchSampler_v2(BatchSampler):
    """
    batch sampler, randomly select n_classes, and n_samples each class
    For training with shuffling
    TO BE USED when n_samples = 4
    """
    def __init__(self, dataset, n_classes, n_samples, n_batches_epoch=None):
        self.labels = dataset.labels
        self.labels = np.array(list(self.labels.values()))
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
        self.new_dataset_size = np.array([len(v) for k, v in self.labels_to_indices.items()]).sum()
        self.n_batches_epoch = n_batches_epoch
        if self.n_batches_epoch is None:
            self.n_batches_epoch = self.new_dataset_size // self.batch_size  # len(self.dataset)
        self.used_label_count = 0
        print('n_batches_epoch', self.n_batches_epoch)

    def __iter__(self):
        self.count = 0
        for i in range(self.n_batches_epoch):
            classes = self.labels_set[self.used_label_count:self.used_label_count + self.n_classes]
            self.used_label_count += self.n_classes
            if self.used_label_count + self.n_classes > len(self.labels_set):
                np.random.shuffle(self.labels_set)
                self.used_label_count = 0
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


class BalanceBatchSampler(BatchSampler):
    """
    batch sampler, randomly select n_classes, and n_samples each class

    TO BE USED when n_samples = 4
    """
    def __init__(self, dataset, n_classes, n_samples, n_batches_epoch=None):
        self.labels = dataset.labels
        self.labels = np.array(list(self.labels.values()))
        self.labels_set = list(set(self.labels))
        self.labels_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        # for l in self.labels_set:
        #     np.random.shuffle(self.labels_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.dataset = dataset
        self.new_dataset_size = np.array([len(v) for k, v in self.labels_to_indices.items()]).sum()
        self.n_batches_epoch = n_batches_epoch
        if self.n_batches_epoch is None:
            self.n_batches_epoch = self.new_dataset_size // self.batch_size  # len(self.dataset)
        self.used_label_count = 0
        print('n_batches_epoch', self.n_batches_epoch)

    def __iter__(self):
        self.count = 0
        for i in range(self.n_batches_epoch):
            classes = self.labels_set[self.used_label_count:self.used_label_count + self.n_classes]
            self.used_label_count += self.n_classes
            if self.used_label_count + self.n_classes > len(self.labels_set):
                # np.random.shuffle(self.labels_set)
                self.used_label_count = 0
            indices = []
            for class_ in classes:
                indices.extend(self.labels_to_indices[class_][self.used_label_indices_count[class_]:
                                                              self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.labels_to_indices[class_]):
                    # np.random.shuffle(self.labels_to_indices[class_])
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


class Reporter(object):
    def __init__(self, ckpt_root, exp, monitor, ckpt_file=None):
        self.ckpt_root = ckpt_root
        self.exp_path = os.path.join(self.ckpt_root, exp)
        self.run_list = os.listdir(self.exp_path)
        self.selected_ckpt = None
        self.selected_epoch = None
        self.selected_log = None
        self.selected_run = None
        self.last_epoch = 0
        self.last_loss = 0
        self.monitor = monitor

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
            if re.search('-1', s):
                matched.remove(s)
            else:
                if self.monitor == 'loss':
                    acc_str = re.search('loss_(.*)\.tar', s).group(1)
                elif self.monitor == 'acc':
                    acc_str = re.search('acc_(.*)\.tar', s).group(1)
                elif self.monitor == 'acc001':
                    acc_str = re.search('acc001_(.*)\.tar', s).group(1)
                loss.append(float(acc_str))

        loss = np.array(loss)
        if self.monitor == 'loss':
            best_idx = np.argmin(loss)
            best_fname = matched[best_idx]
            self.selected_run = best_fname.split(',')[0]
            self.selected_epoch = int(re.search('Epoch_(.*),loss', best_fname).group(1))
        elif self.monitor == 'acc':
            best_idx = np.argmax(loss)
            best_fname = matched[best_idx]
            self.selected_run = best_fname.split(',')[0]
            self.selected_epoch = int(re.search('Epoch_(.*),acc', best_fname).group(1))
        elif self.monitor == 'acc001':
            best_idx = np.argmax(loss)
            best_fname = matched[best_idx]
            self.selected_run = best_fname.split(',')[0]
            self.selected_epoch = int(re.search('Epoch_(.*),acc001', best_fname).group(1))

        ckpt_file = os.path.join(self.exp_path, best_fname)

        self.selected_ckpt = ckpt_file

        return self

    def select_last(self, run=""):

        """
        set self.selected_run, self.selected_ckpt, self.selected_epoch
        :param run:
        :return:
        """
        matched = []
        for fname in self.run_list:
            if fname.startswith(run) and fname.endswith('tar'):
                matched.append(fname)

        import re
        for s in matched:
            if re.search('last_Epoch', s):
                if self.monitor == 'loss':
                    epoch = re.search('last_Epoch_(.*),loss', s).group(1)
                    loss = re.search('loss_(.*)', s).group(1)
                elif self.monitor == 'acc':
                    epoch = re.search('last_Epoch_(.*),acc', s).group(1)
                    loss = re.search('acc_(.*)', s).group(1)
                elif self.monitor == 'acc001':
                    epoch = re.search('last_Epoch_(.*),acc001', s).group(1)
                    loss = re.search('acc001_(.*)', s).group(1)
                last_fname = s

        self.selected_run = last_fname.split(',')[0]
        self.last_epoch = epoch
        self.last_loss = loss

        ckpt_file = os.path.join(self.exp_path, last_fname)
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
