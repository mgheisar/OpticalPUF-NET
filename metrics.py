import numpy as np
import torch


def acc_authentication(target, embedding, n_samples):
    np.random.seed(0)
    Ptp01, Ptp001 = np.zeros(10), np.zeros(10)
    N_enrolled = int(len(target) / n_samples / 2)
    N_H0 = N_enrolled
    target = target.cpu().detach().numpy()
    ids = np.unique(target)
    for rep in range(10):
        data_ids = np.random.choice(len(ids), N_enrolled, replace=False).astype(np.int)
        mask = np.ones(len(ids), np.bool)
        mask[data_ids] = 0
        data_ids = ids[data_ids]
        H0_ids = ids[mask]
        embedding_data, embedding_H1, embedding_H0 = [], [], []
        for i in range(N_enrolled):
            temp = np.where(target == data_ids[i])
            selected_ind = np.random.choice(len(temp[0]), 2, replace=False)

            data_ind = temp[0][selected_ind[0]]
            embedding_data.append(embedding[data_ind.astype(np.int)])

            data_ind = temp[0][selected_ind[1]]
            embedding_H1.append(embedding[data_ind.astype(np.int)])

            temp = np.where(target == H0_ids[i])
            selected_ind = np.random.randint(0, len(temp[0]))
            data_ind = temp[0][selected_ind]
            embedding_H0.append(embedding[data_ind.astype(np.int)])

        embedding_H1 = torch.stack(embedding_H1)
        embedding_H0 = torch.stack(embedding_H0)
        embedding_data = torch.stack(embedding_data)
        H0_claimed_id = np.random.randint(0, N_enrolled, size=N_H0).astype(np.int)
        D00 = torch.norm(embedding_H0 - embedding_data[H0_claimed_id, :], p=2, dim=1).cpu()

        D11 = torch.norm(embedding_H1 - embedding_data, p=2, dim=1).cpu()
        D0 = np.sort(D00)
        D1 = np.sort(D11)

        Pfp = 0.01
        tau = D0[int(Pfp * N_H0)]
        Ptp01[rep] = np.count_nonzero(D1 <= tau) / N_enrolled
        Pfp = 0.001
        tau = D0[int(Pfp * N_H0)]
        Ptp001[rep] = np.count_nonzero(D1 <= tau) / N_enrolled

    Ptp01 = np.mean(Ptp01)
    Ptp001 = np.mean(Ptp001)
    return Ptp01, Ptp001


def shift_img(dataa, shift, n_pixel):
    data_np = dataa.cpu().detach().numpy()
    if shift == 'down':  # ----- Shift n pixel down
        data_np = np.roll(data_np, n_pixel, axis=2)
        data_np[:, :, 0:n_pixel, :] = 0  # data_np[:, :, 1, :]
    elif shift == 'up':  # ----- Shift one pixel up
        data_np = np.roll(data_np, -n_pixel, axis=2)
        data_np[:, :, -n_pixel:, :] = 0
    elif shift == 'right':  # ----- Shift one pixel right
        data_np = np.roll(data_np, n_pixel, axis=3)
        data_np[:, :, :, 0:n_pixel] = 0
    elif shift == 'left':  # ----- Shift one pixel left
        data_np = np.roll(data_np, -n_pixel, axis=3)
        data_np[:, :, :, -n_pixel:] = 0
    data_np = torch.tensor(data_np, dtype=torch.float).to(device)
    return data_np


def l2_norm(x):
    """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-16)
    return x
