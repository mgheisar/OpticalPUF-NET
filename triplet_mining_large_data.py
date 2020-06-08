import torch
import os
import time
from utils import *
from torch.utils.data import DataLoader
import numpy as np
import losses
import models
from checkpoint import CheckPoint
import yaml
import json
import argparse
from history import History
import dill

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(ROOT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  --------------------------------------------------------------------------------------
    # Arguments
    #  --------------------------------------------------------------------------------------
    with open(r'{}/args.yaml'.format(ROOT_DIR)) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    n_classes_train = args_list['n_classes_train']
    n_samples_train = args_list['n_samples_train']
    n_classes_test = args_list['n_classes_test']
    n_samples_test = args_list['n_samples_test']
    emb_dim = args_list['emb_dim']
    model_name = args_list['model_name']
    margin = args_list['margin']
    margin_test = args_list['margin_test']
    soft_margin = args_list['soft_margin']
    triplet_method = args_list['triplet_method']
    lr = args_list['lr']
    n_epoch = args_list['n_epoch']
    run_name = args_list['run_name']
    num_workers = args_list['num_workers']
    N_enrolled = args_list['N_enrolled']
    batch_size_p = args_list['batch_size_p']
    n_save_epoch = args_list['n_save_epoch']
    # Create the parser and Add the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '--model', type=str, default=model_name,
                        help='model name (default: "Resnet50")')
    parser.add_argument('--margin', '--m', type=float, default=margin,
                        help='margin (default: 0.3)')
    parser.add_argument('--margin_test', '--mt', type=float, default=margin_test,
                        help='margin for evaluation (default: 0.3)')
    parser.add_argument('--soft_margin', '--sm', type=int, default=soft_margin,
                        help='apply soft margin? (default: 0)')
    parser.add_argument('--triplet_method', '--tm', type=str, default=triplet_method,
                        help='triplet method (default: "batch_hardv2")')
    parser.add_argument('--run_name', '--rn', type=str, default=run_name,
                        help='The name for this run (default: "Run01-hardv2")')
    parser.add_argument('--num_workers', '--nw', type=int, default=num_workers,
                        help='number of workers for Dataloader (num_workers: 8)')
    parser.add_argument('--start', '--start', type=int, default=0,
                        help='Start from scratch (default: 0)')
    parser.add_argument('--emb_dim', '--dim', type=int, default=emb_dim,
                        help='the embedding dimension of descriptors (dim: 256)')

    args = parser.parse_args()
    model_name = args.model_name
    margin = args.margin
    margin_test = args.margin_test
    soft_margin = args.soft_margin
    triplet_method = args.triplet_method
    run_name = args.run_name
    num_workers = args.num_workers
    start = args.start
    emb_dim = args.emb_dim
    #  --------------------------------------------------------------------------------------
    # Load Train, Validation and Test datasets
    #  --------------------------------------------------------------------------------------
    with open('{}/dataset-puf-all.json'.format(ROOT_DIR), 'r') as fp:
        dataset = json.load(fp)
    partition = dataset['partition']
    labels = dataset['labels']
    labels_train = labels['train']
    labels_validation = labels['validation']
    labels_test = labels['test']
    train_dataset = PairLoader_large(partition['train'], labels_train)
    test_dataset = PairLoader_large(partition['test'], labels_test)
    validation_dataset = PairLoader_large(partition['validation'], labels_validation)
    #  --------------------------------------------------------------------------------------
    # Batch generation P(n_measurement) x K(n_PUF)
    #  --------------------------------------------------------------------------------------
    batch_size = n_classes_train * n_samples_train
    train_batch_sampler = BalanceBatchSampler_large(dataset=train_dataset, n_classes=n_classes_train,
                                                    n_samples=n_samples_train)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)
    validation_batch_sampler = BalanceBatchSampler_large(dataset=validation_dataset, n_classes=n_classes_test,
                                                         n_samples=n_samples_test)
    validation_loader = DataLoader(validation_dataset,
                                   batch_sampler=validation_batch_sampler, num_workers=num_workers)
    test_batch_sampler = BalanceBatchSampler_large(dataset=test_dataset, n_classes=n_classes_test,
                                                   n_samples=n_samples_test)
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=num_workers)
    #  --------------------------------------------------------------------------------------
    # Loss and Model Definitions
    #  --------------------------------------------------------------------------------------
    if triplet_method == "batch_hard":
        loss_fn = losses.BatchHardTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_hardv2":
        loss_fn = losses.BatchHardTripletLoss_v2(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_all":
        loss_fn = losses.BatchAllTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

    model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pretrained=False)
    model.to(device)
    #  --------------------------------------------------------------------------------------
    #  Resume training if start is False
    #  --------------------------------------------------------------------------------------
    if not start:
        last_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                                       exp=triplet_method).select_last(run=run_name).selected_ckpt
        last_epoch = int(Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                                  exp=triplet_method).select_last(run=run_name).last_epoch)
        loss0 = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                         exp=triplet_method).select_last(run=run_name).last_loss
        loss0 = float(loss0[:-4])
        model.load_state_dict(torch.load(last_model_filename)['model_state_dict'])
    else:
        last_epoch = 0
        loss0 = 0

    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
    path_ckpt = '{}/ckpt/{}'.format(ROOT_DIR, triplet_method)
    # learning embedding checkpointer.
    ckpter = CheckPoint(model=model, optimizer=optimizer_model, path=path_ckpt,
                        prefix=run_name, interval=1, save_num=n_save_epoch, loss0=loss0)
    train_hist = History(name='train_hist' + run_name)
    validation_hist = History(name='validation_hist' + run_name)

    #  --------------------------------------------------------------------------------------
    # Preparing data for authentication scenario on train and validation set
    #  --------------------------------------------------------------------------------------
    NqueryH1 = N_enrolled
    NqueryH0 = N_enrolled
    enrolled_loader = {'train': [], 'validation': []}
    H1_loader = {'train': [], 'validation': []}
    H0_loader = {'train': [], 'validation': []}

    for i in range(20):
        # Training set
        partition_x, partition_id = query_partitioning(partition['train'], labels_train,
                                                       N_enrolled, NqueryH0, seed=i)
        enrolled_dataset = PairLoader_large(partition_x['data'], partition_id['data'])
        enrolled_loader['train'].append(
            DataLoader(enrolled_dataset, batch_size=batch_size_p, num_workers=num_workers))
        H1_dataset = PairLoader_large(partition_x['H1'], partition_id['H1'])
        H1_loader['train'].append(DataLoader(H1_dataset, batch_size=batch_size_p, num_workers=num_workers))
        H0_dataset = PairLoader_large(partition_x['H0'], partition_id['H0'])
        H0_loader['train'].append(DataLoader(H0_dataset, batch_size=batch_size_p, num_workers=num_workers))
        # Validation set
        partition_x, partition_id = query_partitioning(partition['validation'], labels_validation,
                                                       N_enrolled, NqueryH0, seed=i)
        enrolled_dataset = PairLoader_large(partition_x['data'], partition_id['data'])
        enrolled_loader['validation'].append(
            DataLoader(enrolled_dataset, batch_size=batch_size_p, num_workers=num_workers))
        H1_dataset = PairLoader_large(partition_x['H1'], partition_id['H1'])
        H1_loader['validation'].append(DataLoader(H1_dataset, batch_size=batch_size_p, num_workers=num_workers))
        H0_dataset = PairLoader_large(partition_x['H0'], partition_id['H0'])
        H0_loader['validation'].append(DataLoader(H0_dataset, batch_size=batch_size_p, num_workers=num_workers))

    #  --------------------------------------------------------------------------------------
    # Computing metrics on validation set before starting training
    #  --------------------------------------------------------------------------------------
    if start:
        # ---------  Validation logs -----------------
        triplet_loss_sum = 0
        num_triplets = 0
        model.eval()
        for batch_idx, (data, target) in enumerate(validation_loader):
            output = model(data)
            loss_outputs = loss_fn(output, target)
            triplet_loss = loss_outputs[0]
            num_hard_triplets = loss_outputs[1]
            triplet_loss_sum += triplet_loss
            num_triplets += num_hard_triplets

        avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
        arg_in = {'N_enrolled': N_enrolled, 'NqueryH1': NqueryH1, 'NqueryH0': NqueryH0, 'emb_dim': emb_dim,
                  'margin_test': margin_test, 'soft_margin': soft_margin}
        validation_logs = {'loss': avg_triplet_loss}
        new_path = ckpter.save(epoch=-1, monitor='loss', loss_acc=validation_logs)
        Ptp01, Ptp001, loss_all_avg, nonzeros = \
            acc_authentication(new_path, enrolled_loader['validation'],
                               H1_loader['validation'], H0_loader['validation'], validation_loader, arg_in)
        validation_logs = {'loss': avg_triplet_loss, 'acc': Ptp01,
                           'loss_all_avg': loss_all_avg, 'acc001': Ptp001, 'nonzeros': nonzeros}
        os.remove(new_path)
        validation_hist.add(logs=validation_logs, epoch=0)
        # ---------  Training logs -----------------
        triplet_loss_sum = 0
        num_triplets = 0
        model.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss_outputs = loss_fn(output, target)
            triplet_loss = loss_outputs[0]
            num_hard_triplets = loss_outputs[1]
            triplet_loss_sum += triplet_loss
            num_triplets += num_hard_triplets

        avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
        arg_in = {'N_enrolled': N_enrolled, 'NqueryH1': NqueryH1, 'NqueryH0': NqueryH0, 'emb_dim': emb_dim,
                  'margin_test': margin_test, 'soft_margin': soft_margin}
        train_logs = {'loss': avg_triplet_loss}
        new_path = ckpter.save(epoch=-1, monitor='loss', loss_acc=train_logs)
        Ptp01, Ptp001, loss_all_avg, nonzeros = \
            acc_authentication(new_path, enrolled_loader['train'],
                               H1_loader['train'], H0_loader['train'], train_loader, arg_in)
        train_logs = {'loss': avg_triplet_loss, 'acc': Ptp01,
                      'loss_all_avg': loss_all_avg, 'acc001': Ptp001, 'nonzeros': nonzeros}
        os.remove(new_path)
        train_hist.add(logs=train_logs, epoch=0)
    else:
        train_hist = dill.load(open(train_hist.name + ".pickle", "rb"))
        train_hist = dill.load(open(train_hist.name + ".pickle", "rb"))
    #  --------------------------------------------------------------------------------------
    # Training
    #  --------------------------------------------------------------------------------------
    for epoch in range(last_epoch, n_epoch):
        epoch_time_start = time.time()
        triplet_loss_sum = 0
        num_triplets = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Apply network to get embeddings
            print('batch', batch_idx)
            output = model(data)
            # Calculating loss
            loss_outputs = loss_fn(output, target)
            triplet_loss = loss_outputs[0]
            num_hard_triplets = loss_outputs[1]
            triplet_loss_sum += triplet_loss
            num_triplets += num_hard_triplets
            # Backward pass
            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()

        avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
        #  --------------------------------------------------------------------------------------
        # Validation History
        #  --------------------------------------------------------------------------------------
        # ---------  Validation logs -----------------
        triplet_loss_sum = 0
        num_triplets = 0
        model.eval()
        for batch_idx, (data, target) in enumerate(validation_loader):
            output = model(data)
            loss_outputs = loss_fn(output, target)
            triplet_loss = loss_outputs[0]
            num_hard_triplets = loss_outputs[1]
            triplet_loss_sum += triplet_loss
            num_triplets += num_hard_triplets

        avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
        arg_in = {'N_enrolled': N_enrolled, 'NqueryH1': NqueryH1, 'NqueryH0': NqueryH0, 'emb_dim': emb_dim,
                  'margin_test': margin_test, 'soft_margin': soft_margin}
        validation_logs = {'loss': avg_triplet_loss}
        new_path = ckpter.save(epoch=-1, monitor='loss', loss_acc=validation_logs)
        Ptp01, Ptp001, loss_all_avg, nonzeros = \
            acc_authentication(new_path, enrolled_loader['validation'],
                               H1_loader['validation'], H0_loader['validation'], validation_loader, arg_in)
        validation_logs = {'loss': avg_triplet_loss, 'acc': Ptp01,
                           'loss_all_avg': loss_all_avg, 'acc001': Ptp001, 'nonzeros': nonzeros}
        os.remove(new_path)
        validation_hist.add(logs=validation_logs, epoch=epoch+1)
        # ---------  Training logs -----------------
        triplet_loss_sum = 0
        num_triplets = 0
        model.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss_outputs = loss_fn(output, target)
            triplet_loss = loss_outputs[0]
            num_hard_triplets = loss_outputs[1]
            triplet_loss_sum += triplet_loss
            num_triplets += num_hard_triplets

        avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
        arg_in = {'N_enrolled': N_enrolled, 'NqueryH1': NqueryH1, 'NqueryH0': NqueryH0, 'emb_dim': emb_dim,
                  'margin_test': margin_test, 'soft_margin': soft_margin}
        train_logs = {'loss': avg_triplet_loss}
        new_path = ckpter.save(epoch=-1, monitor='loss', loss_acc=train_logs)
        Ptp01, Ptp001, loss_all_avg, nonzeros = \
            acc_authentication(new_path, enrolled_loader['train'],
                               H1_loader['train'], H0_loader['train'], train_loader, arg_in)
        train_logs = {'loss': avg_triplet_loss, 'acc': Ptp01,
                      'loss_all_avg': loss_all_avg, 'acc001': Ptp001, 'nonzeros': nonzeros}
        os.remove(new_path)
        train_hist.add(logs=train_logs, epoch=epoch+1)

        epoch_time_end = time.time()
        #  --------------------------------------------------------------------------------------
        # Save last model parameters and check if it is the best
        #  --------------------------------------------------------------------------------------
        if epoch > 0:
            new_path = ckpter.last_delete_and_save(epoch=epoch, monitor='acc', loss_acc=validation_logs)
        if num_triplets:
            ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=validation_logs)
        print(
            'Epoch {}:\tAverage Triplet Loss: {:.4f}\tEpoch Time: {:.3f} hours\tNumber of valid training triplets in epoch: {}'.format(
                epoch + 1,
                avg_triplet_loss,
                (epoch_time_end - epoch_time_start) / 3600,
                num_triplets
            )
        )

    dill.dump(train_hist, file=open("ckpt/"+triplet_method+train_hist.name + ".pickle", "wb"))
    dill.dump(validation_hist, file=open("ckpt/"+triplet_method+validation_hist.name + ".pickle", "wb"))

    # train_hist = dill.load(open("ckpt/"+triplet_method+train_hist.name+".pickle", "rb"))
    # validation_hist = dill.load(open("ckpt/"+triplet_method+validation_hist.name+".pickle", "rb"))
