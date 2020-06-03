from utils import Reporter
import torch
import os
import numpy as np
import models
import yaml
from torch.utils.data import DataLoader
from utils import PairLoader
import json
import time


def main_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    emb_dim = 256
    num_workers = 2

    model_name = "resnet50"  # "resnet50" "vgg16"
    run_name_hardv2 = 'Run05-hardv2'  # 'Run05-all'
    run_name_hard = 'Run05-hard'
    run_name_all = 'Run05-all'

    Pfn05 = {"batch_hardv2": [], "batch_hard": [], "batch_all": []}
    Pfn01 = {"batch_hardv2": [], "batch_hard": [], "batch_all": []}
    Pfn001 = {"batch_hardv2": [], "batch_hard": [], "batch_all": []}
    for t in range(20):
        time_start = time.time()
        N = 2000
        N_enrolled = 1000
        batch_size = 100
        n_samples = 4

        N_chlng = 919
        material = "ZnO"
        thickness_in_nm = [9483, 9563, 9690, 9819, 9925, 9945, 10089, 10137, 10285, 10385, 10485, 10968, 11006, 11056,
                           11071,
                           11093, 11477, 11621, 11623, 11675]
        N_puf = len(thickness_in_nm)
        path_folder = "/nfs/nas4/ID_IOT/ID_IOT/PUF_Data/NEW_Data/Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
        date = "/2019-03-19/Run00/"
        polar = ["Horizontal/hor_", "Horizontal/ver_", "Vertical/hor_", "Vertical/ver_"]
        indices = np.random.choice(range(N_chlng * N_puf), N, replace=False)
        polarization = ["hh", "hv", "vh", "vv"]
        d = 224  # window size
        X = np.zeros((N * n_samples, 3, d, d))
        Y = np.zeros(N * n_samples)
        for i in range(len(indices)):
            ind_puf = int(indices[i] / N_chlng)
            ind_chlng = indices[i] - ind_puf * N_chlng
            pathh = path_folder + material + str(thickness_in_nm[ind_puf]) + date
            for j in range(n_samples):
                data_id = str(thickness_in_nm[ind_puf]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[j]
                X[i * n_samples + j, :, :, :] = np.load(ROOT_DIR + '/data/' + data_id + '.npy')
                Y[i * n_samples + j] = i

        N = N_enrolled
        NqueryH1 = N
        ids = np.unique(Y)
        data_ids = np.random.choice(ids, N, replace=False).astype(np.int)
        # H0_ids = [x for x in ids if x not in enrolled_ids]
        mask = np.ones(len(ids), np.bool)
        mask[data_ids] = 0
        H0_ids = ids[mask]

        data_ind = np.zeros((N, 1))
        data_id = np.zeros((N, 1))
        data_x = np.zeros((N, X.shape[1], X.shape[2], X.shape[3]))

        H1_ind = np.zeros((N, 1))
        H1_id = np.zeros((N, 1))
        H1_x = np.zeros((N, X.shape[1], X.shape[2], X.shape[3]))
        for i in range(N):
            temp = np.where(Y == data_ids[i])
            selected_ind = np.random.choice(len(temp[0]), 2, replace=False)

            data_ind[i] = temp[0][selected_ind[0]]
            data_id[i] = Y[data_ind[i].astype(np.int)]
            data_x[i, :] = X[data_ind[i].astype(np.int), :]

            H1_ind[i] = temp[0][selected_ind[1]]
            H1_id[i] = Y[H1_ind[i].astype(np.int)]
            H1_x[i, :] = X[H1_ind[i].astype(np.int), :]

        NqueryH0 = len(H0_ids)
        H0_ind = np.zeros((NqueryH0, 1))
        H0_id = np.zeros((NqueryH0, 1))
        H0_x = np.zeros((NqueryH0, X.shape[1], X.shape[2], X.shape[3]))
        for i in range(NqueryH0):
            temp = np.where(Y == H0_ids[i])
            selected_ind = np.random.randint(0, len(temp[0]))
            H0_ind[i] = temp[0][selected_ind]
            H0_id[i] = Y[H0_ind[i].astype(np.int)]
            H0_x[i, :] = X[H0_ind[i].astype(np.int), :]

        data_x = torch.tensor(data_x, dtype=torch.float).to(device)
        H1_x = torch.tensor(H1_x, dtype=torch.float).to(device)
        H0_x = torch.tensor(H0_x, dtype=torch.float).to(device)

        dataset = PairLoader(data_x, data_id)
        data_x_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset = PairLoader(H1_x, H1_id)
        H1_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset = PairLoader(H0_x, H0_id)
        H0_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        #  --------------------------------------------------------------------------------------
        # batch_hardv2
        #  --------------------------------------------------------------------------------------
        triplet_method = "batch_hardv2"
        model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pretrained=False)
        model.to(device)
        best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                                       exp=triplet_method).select_best(run=run_name_hardv2).selected_ckpt
        print(best_model_filename)
        model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
        model.eval()
        embedding_data = []
        embedding_H1 = []
        embedding_H0 = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_x_loader):
                embedding = model(data).cpu()
                embedding_data.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H1_loader):
                embedding = model(data).cpu()
                embedding_H1.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H0_loader):
                embedding = model(data).cpu()
                embedding_H0.append(embedding.data.numpy())

        embedding_data = np.array(embedding_data).reshape(-1, emb_dim)
        embedding_H1 = np.array(embedding_H1).reshape(-1, emb_dim)
        embedding_H0 = np.array(embedding_H0).reshape(-1, emb_dim)

        H0_claimed_id = np.random.randint(0, N, size=NqueryH0).astype(np.int)

        D00 = np.linalg.norm(embedding_H0 - embedding_data[H0_claimed_id, :], axis=1)

        xsorted = np.argsort(data_id.squeeze())
        ypos = np.searchsorted(data_id.squeeze()[xsorted], H1_id.squeeze())
        temp = xsorted[ypos]
        D11 = np.linalg.norm(embedding_H1 - embedding_data[temp.astype(np.int), :], axis=1)

        D0 = np.sort(D00)
        D1 = np.sort(D11)

        Pfp = 0.05
        tau = D0[int(Pfp * NqueryH0)]
        Pfn05[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn05', Pfn)

        Pfp = 0.01
        tau = D0[int(Pfp * NqueryH0)]
        Pfn01[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn01', Pfn)

        Pfp = 0.001
        tau = D0[int(Pfp * NqueryH0)]
        Pfn001[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn001', Pfn)

        #  --------------------------------------------------------------------------------------
        # batch_hard
        #  --------------------------------------------------------------------------------------
        triplet_method = "batch_hard"
        model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pretrained=False)
        model.to(device)
        best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                                       exp=triplet_method).select_best(run=run_name_hard).selected_ckpt
        print(best_model_filename)
        model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
        model.eval()
        embedding_data = []
        embedding_H1 = []
        embedding_H0 = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_x_loader):
                embedding = model(data).cpu()
                embedding_data.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H1_loader):
                embedding = model(data).cpu()
                embedding_H1.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H0_loader):
                embedding = model(data).cpu()
                embedding_H0.append(embedding.data.numpy())

        embedding_data = np.array(embedding_data).reshape(-1, emb_dim)
        embedding_H1 = np.array(embedding_H1).reshape(-1, emb_dim)
        embedding_H0 = np.array(embedding_H0).reshape(-1, emb_dim)

        H0_claimed_id = np.random.randint(0, N, size=NqueryH0).astype(np.int)

        D00 = np.linalg.norm(embedding_H0 - embedding_data[H0_claimed_id, :], axis=1)

        xsorted = np.argsort(data_id.squeeze())
        ypos = np.searchsorted(data_id.squeeze()[xsorted], H1_id.squeeze())
        temp = xsorted[ypos]
        D11 = np.linalg.norm(embedding_H1 - embedding_data[temp.astype(np.int), :], axis=1)

        D0 = np.sort(D00)
        D1 = np.sort(D11)

        Pfp = 0.05
        tau = D0[int(Pfp * NqueryH0)]
        Pfn05[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn05', Pfn)

        Pfp = 0.01
        tau = D0[int(Pfp * NqueryH0)]
        Pfn01[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn01', Pfn)

        Pfp = 0.001
        tau = D0[int(Pfp * NqueryH0)]
        Pfn001[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn001', Pfn)

        #  --------------------------------------------------------------------------------------
        # batch_all
        #  --------------------------------------------------------------------------------------
        triplet_method = "batch_all"
        model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pretrained=False)
        model.to(device)
        best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                                       exp=triplet_method).select_best(run=run_name_all).selected_ckpt
        print(best_model_filename)
        model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
        model.eval()
        embedding_data = []
        embedding_H1 = []
        embedding_H0 = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_x_loader):
                embedding = model(data).cpu()
                embedding_data.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H1_loader):
                embedding = model(data).cpu()
                embedding_H1.append(embedding.data.numpy())
            for batch_idx, (data, target) in enumerate(H0_loader):
                embedding = model(data).cpu()
                embedding_H0.append(embedding.data.numpy())

        embedding_data = np.array(embedding_data).reshape(-1, emb_dim)
        embedding_H1 = np.array(embedding_H1).reshape(-1, emb_dim)
        embedding_H0 = np.array(embedding_H0).reshape(-1, emb_dim)

        H0_claimed_id = np.random.randint(0, N, size=NqueryH0).astype(np.int)

        D00 = np.linalg.norm(embedding_H0 - embedding_data[H0_claimed_id, :], axis=1)

        xsorted = np.argsort(data_id.squeeze())
        ypos = np.searchsorted(data_id.squeeze()[xsorted], H1_id.squeeze())
        temp = xsorted[ypos]
        D11 = np.linalg.norm(embedding_H1 - embedding_data[temp.astype(np.int), :], axis=1)

        D0 = np.sort(D00)
        D1 = np.sort(D11)

        Pfp = 0.05
        tau = D0[int(Pfp * NqueryH0)]
        Pfn05[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn05', Pfn)

        Pfp = 0.01
        tau = D0[int(Pfp * NqueryH0)]
        Pfn01[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn01', Pfn)

        Pfp = 0.001
        tau = D0[int(Pfp * NqueryH0)]
        Pfn001[triplet_method].append(1 - np.count_nonzero(D1 <= tau) / NqueryH1)
        # print(triplet_method + ': pfn001', Pfn)

        a_file = open("Pfn05.json", "w")
        json.dump(Pfn05, a_file)
        a_file.close()

        a_file = open("Pfn01.json", "w")
        json.dump(Pfn01, a_file)
        a_file.close()

        a_file = open("Pfn001.json", "w")
        json.dump(Pfn001, a_file)
        a_file.close()
        time_end = time.time()
        print('time : '+str((time_start-time_end)/3600))

    print(np.array(Pfn05['batch_all']))
    print('batch_all_05 : ', np.mean(np.array(Pfn05['batch_all'])))
    print('batch_all_01 : ', np.mean(np.array(Pfn01['batch_all'])))
    print('batch_all_001 : ', np.mean(np.array(Pfn001['batch_all'])))

    print('batch_hard_05 : ', np.mean(np.array(Pfn05['batch_hard'])))
    print('batch_hard_01 : ', np.mean(np.array(Pfn01['batch_hard'])))
    print('batch_hard_001 : ', np.mean(np.array(Pfn001['batch_hard'])))

    print('batch_hardv2_05 : ', np.mean(np.array(Pfn05['batch_hardv2'])))
    print('batch_hardv2_01 : ', np.mean(np.array(Pfn01['batch_hardv2'])))
    print('batch_hardv2_001 : ', np.mean(np.array(Pfn001['batch_hardv2'])))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main_run()
