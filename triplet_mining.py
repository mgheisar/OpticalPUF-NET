import torch
import os
import time
from utils import PairLoader, BalanceBatchSampler, Reporter
from torch.utils.data import DataLoader
import numpy as np
import losses
import models
from checkpoint import CheckPoint
# import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# load data X: input Y: class number
dataset = np.load('dataset-puf.npz')
X = dataset['features']
Y = dataset['labels']
polar = dataset['polar']
n_train = int(0.8 * len(Y))

# Sample data for training and test
Xtrain = X[:n_train]
Xtest = X[n_train:]
ytrain = Y[:n_train]
ytest = Y[n_train:]

Xtrain = np.swapaxes(Xtrain, 1, 3)  # N,1,d,d
Xtest = np.swapaxes(Xtest, 1, 3)
n_train = len(Xtrain)
n_test = len(Xtest)

train_dataset = PairLoader(Xtrain, ytrain)
test_dataset = PairLoader(Xtest, ytest)

# Batch generation P(n_measurement) x K(n_PUF)
n_classes = 16
n_samples = 4
batch_size = n_classes * n_samples
train_batch_sampler = BalanceBatchSampler(dataset=train_dataset, n_classes=n_classes, n_samples=n_samples)
train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
n_classes = 20
n_samples = 4
test_batch_sampler = BalanceBatchSampler(dataset=test_dataset, n_classes=n_classes, n_samples=n_samples)
test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)


model = models.modelTriplet(embedding_dimension=256, pretrained=False)
model.to(device)
# loss_fn = losses.BatchAllTripletLoss(margin=0.3, squared=False, soft_margin=False)
loss_fn = losses.BatchHardTripletLoss(margin=0.3, squared=False, soft_margin=False)
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epoch = 100
run_name = 'Run06'
exp_name = 'batch_hard'
path_ckpt = '{}/ckpt/{}'.format(ROOT_DIR, exp_name)
# train
model.train()

# learning embedding checkpointer.
ckpter = CheckPoint(model=model, optimizer=optimizer_model, path=path_ckpt,
                    prefix=run_name, interval=1, save_num=1)
# metrics = [metrics.AverageNoneZeroTripletsMetric()]
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
        print('batch', batch_idx)

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
best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                               exp=exp_name).select_best(run=run_name).selected_ckpt
model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
model.eval()
print(next(model.parameters()).is_cuda)
batch_all = losses.BatchAllTripletLoss(margin=0.3, squared=False, soft_margin=False)
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
print('acc=', acc, 'num triplets', total_triplets)