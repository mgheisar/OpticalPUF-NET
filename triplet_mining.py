# embedded full size images
import torch
import os
import time
from utils import PairLoader, BalanceBatchSampler, BalanceBatchSampler_v2, Reporter
from metrics import acc_authentication
from torch.utils.data import DataLoader
import losses
import models
from checkpoint import CheckPoint
import numpy as np
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
    np.random.seed(0)
    torch.manual_seed(0)
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
    n_save_epoch = args_list['n_save_epoch']
    data_type = args_list['data']
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
    parser.add_argument('--start', '--start', type=int, default=1,
                        help='Start from scratch (default: 1)')
    parser.add_argument('--emb_dim', '--dim', type=int, default=emb_dim,
                        help='the embedding dimension of descriptors (dim: 256)')
    parser.add_argument('--data_type', '--data', type=str, default=data_type,
                        help='the data source')
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
    data_type = args.data_type
    n_batch_verif = 100
    pooling = False
    # 'data_resizedv2', 'data_c1v2', 'data_c2v2', 'data_complex_resizedv2',
    #                      'data_complex_c1v2', 'data_complex_c2v2'
    # 'data_resized', 'data_c1', 'data_c2', 'data_complex_resized',
    #                      'data_complex_c1', 'data_complex_c2'
    if data_type in ['data_resizedv2', 'data_c1v2', 'data_c2v2', 'data_complex_resizedv2',
                     'data_complex_c1v2', 'data_complex_c2v2']:
        pooling = True
        n_batch_verif = 200
        n_classes_train = 8
        n_classes_test = 8
    print(data_type)
    print('pooling:', pooling)
    #  --------------------------------------------------------------------------------------
    # Load Train, Validation and Test datasets
    #  --------------------------------------------------------------------------------------
    with open('{}/dataset.json'.format(ROOT_DIR), 'r') as fp:
        dataset = json.load(fp)
    partition = dataset['partition']
    labels = dataset['labels']
    train_dataset = PairLoader(partition['train'], labels['train'], data_source=data_type)
    validation_dataset = PairLoader(partition['validation'], labels['validation'], data_source=data_type)
    #  --------------------------------------------------------------------------------------
    # Batch generation P(n_measurement) x K(n_PUF)
    #  --------------------------------------------------------------------------------------
    train_batch_sampler = BalanceBatchSampler_v2(dataset=train_dataset, n_classes=n_classes_train,
                                                 n_samples=n_samples_train)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)
    validation_batch_sampler = BalanceBatchSampler(dataset=validation_dataset, n_classes=n_classes_test,
                                                   n_samples=n_samples_test)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_batch_sampler,
                                   num_workers=num_workers)
    #  --------------------------------------------------------------------------------------
    # Loss and Model Definitions
    #  --------------------------------------------------------------------------------------
    if triplet_method == "batch_hard":
        loss_fn = losses.BatchHardTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_hardv2":
        loss_fn = losses.BatchHardTripletLoss_v2(margin=margin, squared=False, soft_margin=soft_margin)

    elif triplet_method == "batch_all":
        loss_fn = losses.BatchAllTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

    model = models.modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pretrained=False
                                , pooling=pooling)
    model.to(device)
    #  --------------------------------------------------------------------------------------
    #  Resume training if start is False
    #  --------------------------------------------------------------------------------------
    if not start:
        reporter = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                            exp=triplet_method, monitor='acc')
        last_model_filename = reporter.select_last(run=run_name).selected_ckpt
        last_epoch = int(reporter.select_last(run=run_name).last_epoch)
        loss0 = reporter.select_last(run=run_name).last_loss
        loss0 = float(loss0[:-4])
        model.load_state_dict(torch.load(last_model_filename)['model_state_dict'])
    else:
        last_epoch = -1
        loss0 = 0

    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
    path_ckpt = '{}/ckpt/{}'.format(ROOT_DIR, triplet_method)
    # learning embedding checkpointer.
    ckpter = CheckPoint(model=model, optimizer=optimizer_model, path=path_ckpt,
                        prefix=run_name, interval=1, save_num=n_save_epoch, loss0=loss0)
    ckpter_v2 = CheckPoint(model=model, optimizer=optimizer_model, path=path_ckpt,
                           prefix='X'+run_name, interval=1, save_num=n_save_epoch, loss0=loss0)
    train_hist = History(name='train_hist' + run_name)
    validation_hist = History(name='validation_hist' + run_name)
    #  --------------------------------------------------------------------------------------
    # Computing metrics on validation set before starting training
    #  --------------------------------------------------------------------------------------
    if start:
        # ---------  Validation logs -----------------
        print('Computing Validation logs before training')
        batch_all = losses.BatchAllTripletLoss(margin=margin_test, squared=False, soft_margin=soft_margin)
        t = 0
        nonzeros = 0
        triplet_loss_sum = 0
        num_triplets = 0
        model.eval()
        with torch.no_grad():
            n_batches = len(validation_loader)
            Ptp01, Ptp001 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
            emb, tg = [], []
            idx = -1
            for batch_idx, (data, target) in enumerate(validation_loader):
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

        validation_logs = {'loss': avg_triplet_loss, 'acc': np.mean(Ptp01),
                           'loss_all_avg': loss_all_avg, 'acc001': np.mean(Ptp001), 'nonzeros': nonzeros}
        validation_hist.add(logs=validation_logs, epoch=0)
        # ---------  Training logs -----------------
        print('Computing Training logs before training')
        t = 0
        nonzeros = 0
        triplet_loss_sum = 0
        num_triplets = 0
        model.eval()
        with torch.no_grad():
            n_batches = len(train_loader)
            Ptp01, Ptp001 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
            emb, tg = [], []
            idx = -1
            for batch_idx, (data, target) in enumerate(train_loader):
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
                    Ptp01[idx], Ptp001[idx] = acc_authentication(tg, emb, n_samples_train)
                    emb, tg = [], []

            loss_all_avg = 0 if (nonzeros == 0) else t / nonzeros
            avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets

        train_logs = {'loss': avg_triplet_loss, 'acc': np.mean(Ptp01), 'loss_all_avg': loss_all_avg,
                      'acc001': np.mean(Ptp001), 'nonzeros': nonzeros}
        train_hist.add(logs=train_logs, epoch=0)
    else:
        train_hist = dill.load(open("ckpt/" + triplet_method + train_hist.name + ".pickle", "rb"))
        validation_hist = dill.load(open("ckpt/" + triplet_method + validation_hist.name + ".pickle", "rb"))
    #  --------------------------------------------------------------------------------------
    # Training
    #  --------------------------------------------------------------------------------------
    for epoch in range(last_epoch + 1, n_epoch):
        print('Training epoch', epoch + 1)
        # if epoch == 5:
        #     optimizer_model.param_groups[0]['lr'] = lr / 10
        epoch_time_start = time.time()
        triplet_loss_sum = 0
        num_triplets = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Apply network to get embeddings
            embedding = model(data)
            # Calculating loss
            loss_outputs = loss_fn(embedding, target)
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
        print('Computing Validation logs')
        batch_all = losses.BatchAllTripletLoss(margin=margin_test, squared=False, soft_margin=soft_margin)
        t = 0
        nonzeros = 0
        triplet_loss_sum = 0
        num_triplets = 0
        model.eval()
        with torch.no_grad():
            n_batches = len(validation_loader)
            Ptp01, Ptp001 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
            emb, tg = [], []
            idx = -1
            for batch_idx, (data, target) in enumerate(validation_loader):
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
        validation_logs = {'loss': avg_triplet_loss, 'acc': np.mean(Ptp01),
                           'loss_all_avg': loss_all_avg, 'acc001': np.mean(Ptp001), 'nonzeros': nonzeros}
        validation_hist.add(logs=validation_logs, epoch=epoch + 1)
        # ---------  Training logs -----------------
        if (epoch + 1) % 10 == 0:
            print('Computing Training logs')
            t = 0
            nonzeros = 0
            triplet_loss_sum = 0
            num_triplets = 0
            model.eval()
            with torch.no_grad():
                n_batches = len(train_loader)
                Ptp01, Ptp001 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
                emb, tg = [], []
                idx = -1
                for batch_idx, (data, target) in enumerate(train_loader):
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
                        Ptp01[idx], Ptp001[idx] = acc_authentication(tg, emb, n_samples_train)
                        emb, tg = [], []
                loss_all_avg = 0 if (nonzeros == 0) else t / nonzeros
                avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum / num_triplets
            train_logs = {'loss': avg_triplet_loss, 'acc': np.mean(Ptp01), 'loss_all_avg': loss_all_avg,
                          'acc001': np.mean(Ptp001), 'nonzeros': nonzeros}
            train_hist.add(logs=train_logs, epoch=epoch + 1)

        epoch_time_end = time.time()
        #  --------------------------------------------------------------------------------------
        # Save last model parameters and check if it is the best
        #  --------------------------------------------------------------------------------------
        if epoch > 0:
            ckpter.last_delete_and_save(epoch=epoch, monitor='acc', loss_acc=validation_logs)
            ckpter_v2.last_delete_and_save(epoch=epoch, monitor='acc001', loss_acc=validation_logs)
        if num_triplets:
            ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=validation_logs)
            ckpter_v2.check_on(epoch=epoch, monitor='acc001', loss_acc=validation_logs)
        print(
            'Epoch {}:\tAverage Triplet Loss: {:.3f}\tEpoch Time: {:.3f} hours\tNumber of valid training triplets in epoch: {}'.format(
                epoch + 1,
                avg_triplet_loss,
                (epoch_time_end - epoch_time_start) / 3600,
                num_triplets
            )
        )
        dill.dump(train_hist, file=open("ckpt/" + triplet_method + train_hist.name + ".pickle", "wb"))
        dill.dump(validation_hist, file=open("ckpt/" + triplet_method + validation_hist.name + ".pickle", "wb"))
