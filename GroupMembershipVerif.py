import numpy as np
from utils_group import baseline_group_representation, partition_data, hashing
from scipy.stats import zscore
import json
import os

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    with open('{}/data_puf.json'.format(ROOT_DIR), 'r') as fp:
        dataset_t = json.load(fp)
    dataset = dataset_t[0]
    np.random.seed(0)
    dim = len(dataset['data_x'][0])
    data_x = zscore(np.stack(dataset['data_x']), axis=1)  # N*d
    # data_id = np.stack(dataset['data_id'])
    H1_x = zscore(np.stack(dataset['H1_x']), axis=1)
    # H1_id = np.stack(dataset['H1_id'])
    H0_x = zscore(np.stack(dataset['H0_x']), axis=1)
    # H0_id = np.stack(dataset['H0_id'])
    n_q0 = H0_x.shape[0]
    n_q1 = H1_x.shape[0]

    l = int(dim)
    Ptp01, Ptp05 = [], []
    S_x = int(l * 0.7)  # 610
    group_member = 4
    W = np.random.random([dim, l])
    U, S, V = np.linalg.svd(W)
    W = U[:, :l]
    # mat = W.T @ W
    param = {'method': 'EoA', 'agg': 'sum', 'W': W, 'S_x': S_x}
    # Assign data to groups
    groups = partition_data(data_x, group_member, partitioning='random')
    # Compute group representations
    group_vec = baseline_group_representation(groups, param)
    group_vec = np.array(group_vec)
    # The embedding for H0 queries
    Q0 = hashing(H0_x.T, param['W'], param['S_x'])
    H0_claimed_id = np.random.randint(0, len(groups['ind']), size=n_q0).astype(np.int)
    D00 = np.linalg.norm(Q0 - group_vec[:, H0_claimed_id], axis=0)
    # The embedding for H1 queries
    H1_group_id = np.zeros(n_q1)
    Q1 = hashing(H1_x.T, param['W'], param['S_x'])
    # Determine the group identity of H1 queries
    for i in range(len(groups['ind'])):
        group_id = [dataset['data_id'][x] for x in groups['ind'][i]]
        a = [n for n, x in enumerate(dataset['H1_id']) for y in group_id if x == y]
        for x in a:
            H1_group_id[x] = i

    D11 = np.linalg.norm(Q1 - group_vec[:, H1_group_id.astype(np.int)], axis=0)
    D0 = np.sort(D00)
    D1 = np.sort(D11)

    Pfp = 0.01
    tau = D0[int(Pfp * n_q0)]
    Ptp01.append(np.count_nonzero(D1 <= tau) / n_q1)
    Pfp = 0.05
    tau = D0[int(Pfp * n_q0)]
    Ptp05.append(np.count_nonzero(D1 <= tau) / n_q1)
    print('Ptp01:', Ptp01, 'Ptp05', Ptp05)
