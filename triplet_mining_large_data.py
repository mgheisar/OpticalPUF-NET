import torch
import os
import time
from .utils import PairLoader_large, BalanceBatchSampler_large, Reporter
from torch.utils.data import DataLoader
import numpy as np
from .losses import *
from .models import *
from .checkpoint import CheckPoint
import yaml
import json
# import metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(r'args.yaml') as file:
    args_list = yaml.load(file, Loader=yaml.FullLoader)

train_ratio = args_list['train_ratio']
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# load data X: input Y: class number
with open('/nfs/nas4/marzieh/marzieh/puf-featureextraction/dataset-puf-all.json', 'r') as fp:
    dataset = json.load(fp)

partition = dataset['partition']
labels = dataset['labels']
labels_train = labels['train']
labels_test = labels['test']
train_dataset = PairLoader_large(partition['train'], labels_train)
test_dataset = PairLoader_large(partition['test'], labels_test)

# Batch generation P(n_measurement) x K(n_PUF)
batch_size = n_classes_train * n_samples_train
train_batch_sampler = BalanceBatchSampler_large(dataset=train_dataset, n_classes=n_classes_train, n_samples=n_samples_train)
train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)
test_batch_sampler = BalanceBatchSampler_large(dataset=test_dataset, n_classes=n_classes_test, n_samples=n_samples_test)
test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=num_workers)

model = modelTriplet(embedding_dimension=emb_dim, model_architecture=model_name, pretrained=False)
model.to(device)

if triplet_method == "batch_hard":
    loss_fn = BatchHardTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

elif triplet_method == "batch_hardv2":
    loss_fn = BatchHardTripletLoss_v2(margin=margin, squared=False, soft_margin=soft_margin)

elif triplet_method == "batch_all":
    loss_fn = BatchAllTripletLoss(margin=margin, squared=False, soft_margin=soft_margin)

optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)

path_ckpt = '{}/ckpt/{}'.format(ROOT_DIR, triplet_method)
# train
model.train()

# learning embedding checkpointer.
ckpter = CheckPoint(model=model, optimizer=optimizer_model, path=path_ckpt,
                    prefix=run_name, interval=1, save_num=1)
# metrics = [metrics.AverageNoneZeroTripletsMetric()]
model.eval()
batch_all = BatchAllTripletLoss(margin=margin_test, squared=False, soft_margin=soft_margin)

t = 0
total_triplets_train_0 = 0
loss_test = 0
anchor_embeddings = []
negative_embeddings = []
positive_embeddings = []
anchor_distances = []

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        print('batch', batch_idx)
        outputs = model(data)
        batch_all_outputs = batch_all(outputs, target)
        t += int(batch_all_outputs[1])
        total_triplets_train_0 += int(batch_all_outputs[2])
        loss_test += batch_all_outputs[0]
    acc_train_0 = 1 - t / total_triplets_train_0
print('acc on train before training=', acc_train_0, 'num triplets', total_triplets_train_0)

t = 0
total_triplets_test_0 = 0
loss_test = 0
anchor_embeddings = []
negative_embeddings = []
positive_embeddings = []
anchor_distances = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        print('batch', batch_idx)
        outputs = model(data)
        batch_all_outputs = batch_all(outputs, target)
        t += int(batch_all_outputs[1])
        total_triplets_test_0 += int(batch_all_outputs[2])
        loss_test += batch_all_outputs[0]
    acc_test_0 = 1 - t / total_triplets_test_0
print('acc on test before training=', acc_test_0, 'num triplets', total_triplets_test_0)
for epoch in range(n_epoch):
    # for metric in metrics:
    #     metric.reset()
    epoch_time_start = time.time()
    triplet_loss_sum = 0
    num_triplets = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Apply network to get embeddings
        # print(data.is_cuda)  # ----------------------------------------------------------------------
        # print(data.size())
        print('batch', batch_idx)
        output = model(data)
        # Calculating loss
        loss_outputs = loss_fn(output, target)
        triplet_loss = loss_outputs[0]
        num_hard_triplets = loss_outputs[1]
        # for metric in metrics:
        #     metric(output, target, loss_outputs)
        triplet_loss_sum += triplet_loss
        num_triplets += num_hard_triplets
        # Backward pass
        optimizer_model.zero_grad()
        triplet_loss.backward()
        # print(triplet_loss.is_cuda)  # ------------------------------------------------------------
        optimizer_model.step()

    avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum  # / num_triplets
    epoch_time_end = time.time()
    train_logs = {'loss': avg_triplet_loss}
    # for metric in metrics:
    #     train_logs[metric.name()] = metric.value()
    # train_hist.add(logs=train_logs, epoch=epoch)
    if num_triplets:
        ckpter.check_on(epoch=epoch, monitor='loss', loss_acc=train_logs)

    print(
        'Epoch {}:\tAverage Triplet Loss: {:.4f}\tEpoch Time: {:.3f} hours\tNumber of valid training triplets in epoch: {}'.format(
            epoch + 1,
            avg_triplet_loss,
            (epoch_time_end - epoch_time_start) / 3600,
            num_triplets
        )
    )
#  --------------------------------------------------------------------------------------
# Evaluation
#  --------------------------------------------------------------------------------------
print('acc on train before training=', acc_train_0, 'num triplets', total_triplets_train_0)
print('acc on test before training=', acc_test_0, 'num triplets', total_triplets_test_0)

best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                               exp=triplet_method).select_best(run=run_name).selected_ckpt
model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
model.eval()
batch_all = BatchAllTripletLoss(margin=margin_test, squared=False, soft_margin=soft_margin)

t = 0
total_triplets = 0
anchor_embeddings = []
negative_embeddings = []
positive_embeddings = []
anchor_distances = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        outputs = model(data)
        batch_all_outputs = batch_all(outputs, target)
        t += int(batch_all_outputs[1])
        total_triplets += int(batch_all_outputs[2])
    acc = 1 - t / total_triplets
print('acc on training set =', acc, 'num triplets', total_triplets)


t = 0
total_triplets = 0
anchor_embeddings = []
negative_embeddings = []
positive_embeddings = []
anchor_distances = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        outputs = model(data)
        batch_all_outputs = batch_all(outputs, target)
        t += int(batch_all_outputs[1])
        total_triplets += int(batch_all_outputs[2])
    acc = 1 - t / total_triplets
print('acc on test =', acc, 'num triplets', total_triplets)
