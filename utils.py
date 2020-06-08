import numpy as np
from torch.utils import data
# import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler
import torch
import os
import models
import losses
# from sklearn.manifold import TSNE
# asc_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
#           '#00ff7f', '#9400d3', '#3b3b3b', '#0000ee', '#bcd2ee']
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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
        Xtensor = torch.tensor(np.load(ROOT_DIR+'/data/'
                                       + ID + '.npy'), dtype=torch.float).to(device)
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
        self.last_epoch = 0
        self.last_loss = 0

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
                epoch = re.search('last_Epoch_(.*),loss', s).group(1)
                loss = re.search('loss_(.*)', s).group(1)
                last_fname = s

        self.selected_run = last_fname.split(',')[0]
        self.last_epoch = epoch
        self.last_loss = loss

        ckpt_file = os.path.join(self.exp_path, last_fname)
        self.selected_ckpt = ckpt_file

        return self


def query_partitioning(data_id, labels, N_enrolled, NqueryH0, seed):
    np.random.seed(seed)
    NqueryH1 = N_enrolled
    Y = np.array(list(labels.values()))
    ids = np.unique(Y)
    data_ids = np.random.choice(len(ids), N_enrolled, replace=False).astype(np.int)
    mask = np.ones(len(ids), np.bool)
    mask[data_ids] = 0
    data_ids = ids[data_ids]
    Non_H1_ids = ids[mask]
    temp = np.random.choice(len(Non_H1_ids), NqueryH0, replace=False).astype(np.int)
    H0_ids = Non_H1_ids[temp]

    partition_x = {'data': [], 'H1': [], 'H0': []}
    partition_id = {'data': {}, 'H1': {}, 'H0': {}}
    for i in range(N_enrolled):
        temp = np.where(Y == data_ids[i])
        selected_ind = np.random.choice(len(temp[0]), 2, replace=False)

        data_ind = temp[0][selected_ind[0]]
        partition_x['data'].append(data_id[data_ind])
        partition_id['data'][data_id[data_ind]] = (Y[data_ind.astype(np.int)])

        data_ind = temp[0][selected_ind[1]]
        partition_x['H1'].append(data_id[data_ind])
        partition_id['H1'][data_id[data_ind]] = Y[data_ind.astype(np.int)]

    for i in range(NqueryH0):
        temp = np.where(Y == H0_ids[i])
        selected_ind = np.random.randint(0, len(temp[0]))
        data_ind = temp[0][selected_ind]
        partition_x['H0'].append(data_id[data_ind])
        partition_id['H0'][data_id[data_ind]] = (Y[data_ind.astype(np.int)])
    return partition_x, partition_id


def acc_authentication(model_filename, enrolled_loader, H1_loader, H0_loader, data_loader, arg_in):
    model = models.modelTriplet()
    model.to(device)
    model.load_state_dict(torch.load(model_filename)['model_state_dict'])
    model.eval()
    Ptp01, Ptp001 = np.zeros(20), np.zeros(20)
    for i in range(len(enrolled_loader)):
        embedding_data = []
        embedding_H1 = []
        embedding_H0 = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(enrolled_loader[i]):
                embedding = model(data).cpu()
                embedding_data.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H1_loader[i]):
                embedding = model(data).cpu()
                embedding_H1.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H0_loader[i]):
                embedding = model(data).cpu()
                embedding_H0.append(embedding.data.numpy())

        embedding_data = np.vstack(embedding_data).reshape(-1, arg_in['emb_dim'])
        embedding_H1 = np.vstack(embedding_H1).reshape(-1, arg_in['emb_dim'])
        embedding_H0 = np.vstack(embedding_H0).reshape(-1, arg_in['emb_dim'])

        H0_claimed_id = np.random.randint(0, arg_in['N_enrolled'], size=arg_in['NqueryH0']).astype(np.int)
        D00 = np.linalg.norm(embedding_H0 - embedding_data[H0_claimed_id, :], axis=1)

        # xsorted = np.argsort(partition_id['data'].values().squeeze())
        # ypos = np.searchsorted(partition_id['data'].squeeze()[xsorted], partition_id['H1'].squeeze())
        # temp = xsorted[ypos]
        # D11 = np.linalg.norm(embedding_H1 - embedding_data[temp.astype(np.int), :], axis=1)

        D11 = np.linalg.norm(embedding_H1 - embedding_data, axis=1)
        D0 = np.sort(D00)
        D1 = np.sort(D11)

        Pfp = 0.01
        tau = D0[int(Pfp * arg_in['NqueryH0'])]
        Ptp01[i] = np.count_nonzero(D1 <= tau) / arg_in['NqueryH1']
        Pfp = 0.01
        tau = D0[int(Pfp * arg_in['NqueryH0'])]
        Ptp001[i] = np.count_nonzero(D1 <= tau) / arg_in['NqueryH1']

    Ptp01 = np.mean(Ptp01)
    Ptp001 = np.mean(Ptp001)

    batch_all = losses.BatchAllTripletLoss(margin=arg_in['margin_test'], squared=False, soft_margin=arg_in['soft_margin'])
    t = 0
    total_triplets = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            outputs = model(data)
            batch_all_outputs = batch_all(outputs, target)
            t += int(batch_all_outputs[1])
            total_triplets += int(batch_all_outputs[2])
        loss_all_avg = t / total_triplets
        loss_total_triplet = total_triplets
    return Ptp01, Ptp001, loss_all_avg, loss_total_triplet

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
