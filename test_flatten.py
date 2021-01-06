# flatten images
import torch
import os
from utils import PairLoader, BalanceBatchSampler
from torch.utils.data import DataLoader
import yaml
import json
import numpy as np


def acc_authentication(target_, embedding_, n_samples):
    np.random.seed(0)
    Ptp01_, Ptp001_ = np.zeros(10), np.zeros(10)
    N_enrolled = int(len(target_) / n_samples / 2)
    N_H0 = N_enrolled
    ids = np.unique(target_)
    for rep in range(10):
        data_ids = np.random.choice(len(ids), N_enrolled, replace=False).astype(np.int)
        mask = np.ones(len(ids), np.bool)
        mask[data_ids] = 0
        data_ids = ids[data_ids]
        H0_ids = ids[mask]
        embedding_data, embedding_H1, embedding_H0 = [], [], []
        for i in range(N_enrolled):
            temp = np.where(target_ == data_ids[i])
            selected_ind = np.random.choice(len(temp[0]), 2, replace=False)

            data_ind = temp[0][selected_ind[0]]
            embedding_data.append(embedding_[data_ind.astype(np.int)])

            data_ind = temp[0][selected_ind[1]]
            embedding_H1.append(embedding_[data_ind.astype(np.int)])

            temp = np.where(target_ == H0_ids[i])
            selected_ind = np.random.randint(0, len(temp[0]))
            data_ind = temp[0][selected_ind]
            embedding_H0.append(embedding_[data_ind.astype(np.int)])

        embedding_H1 = np.stack(embedding_H1)
        embedding_H0 = np.stack(embedding_H0).squeeze()
        embedding_data = np.stack(embedding_data).squeeze()
        H0_claimed_id = np.random.randint(0, N_enrolled, size=N_H0).astype(np.int)
        D00 = np.linalg.norm(embedding_H0 - embedding_data[H0_claimed_id, :], axis=1)

        D11 = np.linalg.norm(embedding_H1 - embedding_data, axis=1)
        D0 = np.sort(D00)
        D1 = np.sort(D11)

        Pfp = 0.01
        tau = D0[int(Pfp * N_H0)]
        Ptp01_[rep] = np.count_nonzero(D1 <= tau) / N_enrolled
        Pfp = 0.001
        tau = D0[int(Pfp * N_H0)]
        Ptp001_[rep] = np.count_nonzero(D1 <= tau) / N_enrolled

    Ptp01_ = np.mean(Ptp01_)
    Ptp001_ = np.mean(Ptp001_)
    return Ptp01_, Ptp001_


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    device = 'cpu'
    np.random.seed(0)
    with open(r'{}/args.yaml'.format(ROOT_DIR)) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    num_workers = args_list['num_workers']
    n_classes_test = args_list['n_classes_test']
    n_samples_test = args_list['n_samples_test']
    n_batch_verif = 114
    # 'data_c1','data_c2','data_complex_c1','data_complex_c2','data_resized','data_complex_resized'
    # 'data_c1v2','data_c2v2','data_complex_c1v2','data_complex_c2v2','data_resizedv2','data_complex_resizedv2'

    data_vec = ['data_resizedv2', 'data_c1v2', 'data_c2v2', 'data_complex_resizedv2', 'data_complex_c1v2', 'data_complex_c2v2']
    log_acc = []
    for i_d in range(6):
        data_type = data_vec[i_d]
        with open('{}/dataset.json'.format(ROOT_DIR), 'r') as fp:
            dataset = json.load(fp)
        partition = dataset['partition']
        labels = dataset['labels']
        test_dataset = PairLoader(partition['test'], labels['test'], data_source=data_type)

        test_batch_sampler = BalanceBatchSampler(dataset=test_dataset, n_classes=n_classes_test,
                                                 n_samples=n_samples_test)
        test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=num_workers)
        n_batches = len(test_loader)
        Ptp01, Ptp001 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
        emb, tg = [], []
        idx = -1
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target.cpu().numpy()
            data = data[:, 0, :, :]
            data = data.cpu().numpy().reshape(data.shape[0], -1)
            embedding = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-16)
            emb.append(embedding)
            tg.append(target)
            if (batch_idx + 1) % n_batch_verif == 0:
                idx += 1
                emb = np.stack(emb).reshape(-1, data.shape[-1])
                tg = np.stack(tg).flatten()
                Ptp01[idx], Ptp001[idx] = acc_authentication(tg, emb, n_samples_test)
                emb, tg = [], []
        test_log = {'acc': np.mean(Ptp01), 'acc001': np.mean(Ptp001)}
        print(test_log)
        log_acc.append(test_log)
    print(log_acc)


#  result : {'acc': 0.04225, 'acc001': 0.011875}
