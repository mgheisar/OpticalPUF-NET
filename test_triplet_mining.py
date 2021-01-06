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

if __name__ == '__main__':
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
    soft_margin = args_list['soft_margin']
    margin = args_list['margin']
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
    n_batch_verif = 114
    pooling = False
    with open('{}/dataset.json'.format(ROOT_DIR), 'r') as fp:  # dataset_complex_full_resized
        dataset = json.load(fp)
    partition = dataset['partition']
    labels = dataset['labels']
    # 'data_c1','data_c2','data_complex_c1','data_complex_c2','data_resized','data_complex_resized'
    # 'data_c1v2','data_c2v2','data_complex_c1v2','data_complex_c2v2','data_resizedv2','data_complex_resizedv2'

    if data_type in ['data_resizedv2', 'data_complex_resizedv2', 'data_c1v2', 'data_complex_c1v2',
                     'data_c2v2', 'data_complex_c2v2']:
        pooling = True
        n_batch_verif = 229
        n_classes_test = 8

    # data_x = partition['train'] + partition['validation'] + partition['test']
    # data_y = {**labels['train'], **labels['validation'], **labels['test']}
    data_x = partition['test']
    data_y = labels['test']
    test_dataset = PairLoader(data_x, data_y, data_source=data_type)
    test_batch_sampler = BalanceBatchSampler(dataset=test_dataset, n_classes=n_classes_test,
                                             n_samples=n_samples_test)
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler,
                             num_workers=num_workers)
    if triplet_method == "batch_hard":
        loss_fn = losses.BatchHardTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_hardv2":
        loss_fn = losses.BatchHardTripletLoss_v2(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_all":
        loss_fn = losses.BatchAllTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)
    # rt = '/nfs/nas4/marzieh/marzieh/puf/ckpt/batch_hardv2/'
    # model_filename = rt + 'Run004,modelTriplet,Epoch_345,acc_0.999688.tar'
    # model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
    #                           exp=triplet_method, monitor='acc').select_best(run=run_name).selected_ckpt

    model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                              exp=triplet_method, monitor='acc001').select_best(run='X' + run_name).selected_ckpt
    print(model_filename)
    model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pooling=pooling)
    model.to(device)
    model.load_state_dict(torch.load(model_filename)['model_state_dict'])

    print('Evaluating model on test data')
    batch_all = losses.BatchAllTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)
    t = 0
    nonzeros = 0
    triplet_loss_sum = 0
    num_triplets = 0
    model.eval()
    with torch.no_grad():
        n_batches = len(test_loader)
        Ptp01, Ptp001 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
        emb, tg = [], []
        idx = -1
        for batch_idx, (data, target) in enumerate(test_loader):
            embedding = model(data)
            loss_outputs = loss_fn(embedding, target)
            triplet_loss = loss_outputs[0]
            num_hard_triplets = loss_outputs[1]
            triplet_loss_sum += triplet_loss
            num_triplets += num_hard_triplets

            batch_all_outputs = batch_all(embedding, target)
            t += int(batch_all_outputs[1])
            nonzeros += int(batch_all_outputs[2])

            emb.append(embedding)
            tg.append(target)
            if (batch_idx + 1) % n_batch_verif == 0:
                idx += 1
                emb = torch.stack(emb).flatten(start_dim=0, end_dim=1)
                tg = torch.stack(tg).flatten(start_dim=0, end_dim=1)
                Ptp01[idx], Ptp001[idx] = acc_authentication(tg, emb, n_samples_test)
                emb, tg = [], []

        loss_all_avg = 0 if (nonzeros == 0) else t / nonzeros
        avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
    test_log = {'acc': np.mean(Ptp01), 'acc001': np.mean(Ptp001)}
    print(test_log)

    if data_type in ['data_resizedv2', 'data_complex_resizedv2', 'data_c1v2', 'data_complex_c1v2',
                     'data_c2v2', 'data_complex_c2v2']:
        pooling = True
        n_batch_verif = 200
        n_classes_test = 8

    if triplet_method == "batch_hard":
        loss_fn = losses.BatchHardTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_hardv2":
        loss_fn = losses.BatchHardTripletLoss_v2(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_all":
        loss_fn = losses.BatchAllTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)
    # rt = '/nfs/nas4/marzieh/marzieh/puf/ckpt/batch_hardv2/'
    # model_filename = rt + 'Run004,modelTriplet,Epoch_345,acc_0.999688.tar'
    # model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
    #                           exp=triplet_method, monitor='acc').select_best(run=run_name).selected_ckpt

    model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                              exp=triplet_method, monitor='acc001').select_best(run='X' + run_name).selected_ckpt
    print(model_filename)
    model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pooling=pooling)
    model.to(device)
    model.load_state_dict(torch.load(model_filename)['model_state_dict'])

    print('Evaluating model on test data')
    batch_all = losses.BatchAllTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)
    t = 0
    nonzeros = 0
    triplet_loss_sum = 0
    num_triplets = 0
    model.eval()
    with torch.no_grad():
        n_batches = len(test_loader)
        Ptp01, Ptp001 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
        emb, tg = [], []
        idx = -1
        for batch_idx, (data, target) in enumerate(test_loader):
            embedding = model(data)
            loss_outputs = loss_fn(embedding, target)
            triplet_loss = loss_outputs[0]
            num_hard_triplets = loss_outputs[1]
            triplet_loss_sum += triplet_loss
            num_triplets += num_hard_triplets

            batch_all_outputs = batch_all(embedding, target)
            t += int(batch_all_outputs[1])
            nonzeros += int(batch_all_outputs[2])

            emb.append(embedding)
            tg.append(target)
            if (batch_idx + 1) % n_batch_verif == 0:
                idx += 1
                emb = torch.stack(emb).flatten(start_dim=0, end_dim=1)
                tg = torch.stack(tg).flatten(start_dim=0, end_dim=1)
                Ptp01[idx], Ptp001[idx] = acc_authentication(tg, emb, n_samples_test)
                emb, tg = [], []

        loss_all_avg = 0 if (nonzeros == 0) else t / nonzeros
        avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
    test_log = {'acc': np.mean(Ptp01), 'acc001': np.mean(Ptp001)}
    print(test_log)
