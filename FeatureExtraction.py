# embedded images with first cropping method
import torch
import os
from utils import PairLoader, Reporter, BalanceBatchSampler
from torch.utils.data import DataLoader
import yaml
import json
import argparse
import models
from metrics import acc_authentication
import losses
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def extract_features():
    torch.multiprocessing.set_start_method('spawn', force=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(r'{}/args.yaml'.format(ROOT_DIR)) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    emb_dim = args_list['emb_dim']
    model_name = args_list['model_name']
    triplet_method = args_list['triplet_method']
    run_name = args_list['run_name']
    num_workers = args_list['num_workers']
    n_classes_test = args_list['n_classes_test']
    n_samples_test = args_list['n_samples_test']
    data_type = args_list['data']

    parser = argparse.ArgumentParser()
    parser.add_argument('--triplet_method', '--tm', type=str, default=triplet_method,
                        help='triplet method (default: "batch_hard")')
    parser.add_argument('--run_name', '--rn', type=str, default=run_name,
                        help='The name for this run (default: "Run01-hardv2")')
    parser.add_argument('--data_type', '--data', type=str, default=data_type,
                        help='the data source')

    args = parser.parse_args()
    triplet_method = args.triplet_method
    run_name = args.run_name
    data_type = args.data_type
    pooling = False
    with open('{}/dataset.json'.format(ROOT_DIR), 'r') as fp:
        dataset = json.load(fp)
    partition = dataset['partition']
    labels = dataset['labels']
    # 'data_c1','data_c2','data_complex_c1','data_complex_c2','data_resized','data_complex_resized'
    # 'data_c1v2','data_c2v2','data_complex_c1v2','data_complex_c2v2','data_resizedv2','data_complex_resizedv2'

    if data_type in ['data_resizedv2', 'data_complex_resizedv2', 'data_c1v2', 'data_complex_c1v2',
                     'data_c2v2', 'data_complex_c2v2']:
        pooling = True
        n_classes_test = 8
    data_x = partition['train'] + partition['validation'] + partition['test']
    data_y = {**labels['train'], **labels['validation'], **labels['test']}
    test_dataset = PairLoader(data_x, data_y, data_source=data_type)
    test_batch_sampler = BalanceBatchSampler(dataset=test_dataset, n_classes=n_classes_test,
                                             n_samples=n_samples_test)
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler,
                             num_workers=num_workers)
    model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                              exp=triplet_method, monitor='acc').select_best(run=run_name).selected_ckpt
    print(model_filename)
    model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pooling=pooling)
    model.to(device)
    model.load_state_dict(torch.load(model_filename)['model_state_dict'])
    model.eval()
    with torch.no_grad():
        emb, tg = [], []
        for batch_idx, (data, target) in enumerate(test_loader):
            embedding = model(data).cpu().numpy()
            target = target.cpu().numpy()
            emb.append(embedding)
            tg.append(target)

        features = {'emb': emb, 'tg': tg}
        with open('features_1.json', 'w') as f_out:
            json.dump(features, f_out, cls=NumpyEncoder)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    extract_features()
    with open('{}/features_1.json'.format(ROOT_DIR), 'r') as fp:
        dataset = json.load(fp)
    np.random.seed(0)
    dim = len(dataset['emb'][0][0])
    feat = np.stack(dataset['emb']).reshape(-1, dim)
    labels = np.stack(dataset['tg']).reshape(-1)
    n_samples = 4
    N_enrolled = 400  # 9180 int(len(labels) / n_samples / 2):9188
    N_H0 = N_enrolled
    ids = np.unique(labels)
    data_query = []
    for ii in range(10):
        data_ids = np.random.choice(len(ids), N_enrolled, replace=False).astype(np.int)
        mask = np.ones(len(ids), np.bool)
        mask[data_ids] = 0
        data_ids = ids[data_ids]
        H0_ids = ids[mask]
        data_x, H1_x, H0_x = np.zeros([N_enrolled, dim]), np.zeros([N_enrolled, dim]), np.zeros([N_enrolled, dim])
        data_ind, H1_ind, H0_ind = np.zeros(N_enrolled), np.zeros(N_enrolled), np.zeros(N_enrolled)
        data_id, H1_id, H0_id = np.zeros(N_enrolled), np.zeros(N_enrolled), np.zeros(N_enrolled)
        for i in range(N_enrolled):
            temp = np.where(labels == data_ids[i])
            selected_ind = np.random.choice(len(temp[0]), 2, replace=False)
            data_id[i] = data_ids[i]

            data_ind[i] = temp[0][selected_ind[0]]
            data_x[i] = feat[data_ind[i].astype(np.int)]

            H1_ind[i] = temp[0][selected_ind[1]]
            H1_x[i] = feat[H1_ind[i].astype(np.int)]

            temp = np.where(labels == H0_ids[i])
            selected_ind = np.random.randint(0, len(temp[0]))
            H0_ind[i] = temp[0][selected_ind]
            H0_x[i] = feat[H0_ind[i].astype(np.int)]
            H0_id[i] = H0_ids[i]

        data_query.append({'data_x': data_x, 'H1_x': H1_x, 'H0_x': H0_x, 'data_id': data_id,
                           'H1_id': data_id, 'H0_id': H0_id, 'data_ind': data_ind,
                           'H1_ind': H1_ind, 'H0_ind': H0_ind})
    with open('data_puf_n1.json', 'w') as f_out:
        json.dump(data_query, f_out, cls=NumpyEncoder)
