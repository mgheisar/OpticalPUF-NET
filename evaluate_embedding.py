import numpy as np
# from utils import preprocess
import torch
import time
from utils import PairLoader, BalanceBatchSampler, Reporter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from losses import *
from models import modelTriplet
from checkpoint import *
from metrics import *
from sklearn.decomposition import PCA
# torch.cuda.set_device(0)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# load data X: input Y: class number
dataset = np.load('dataset-puf.npz')
X = dataset['features']
Y = dataset['labels']
polar = dataset['polar']
n_train = int(0.4 * len(Y))

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


model = modelTriplet(embedding_dimension=128, pretrained=False)
# # model = model.cuda()
# loss_fn = BatchHardTripletLoss(margin=0.3, squared=False, soft_margin=False)
# # loss_fn = BatchHardTripletLoss(margin=0.3, squared=False, soft_margin=True)
# optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)
# n_epoch = 100
#
# # # learning embedding checkpointer.
# # ckpter = CheckPoint(model=model, optimizer=optimizer_model,
# #                     path='{}/ckpt/batch_hard'.format(ROOT_DIR),
# #                     prefix='Run04', interval=1, save_num=1)
# metrics = [AverageNoneZeroTripletsMetric()]
# for epoch in range(n_epoch):
#     model.eval()
#     t = 0
#     total_triplets = 0
#     anchor_embeddings = []
#     negative_embeddings = []
#     positive_embeddings = []
#     anchor_distances = []
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             if batch_idx > 3:
#                 continue
#             num_triplets = len(target)
#             for ind in range(num_triplets):
#                 classes = np.unique(target)
#                 pos_class = np.random.choice(classes)
#                 neg_class = np.random.choice(classes)
#
#                 while len(target == pos_class) < 2:
#                     pos_class = np.random.choice(classes)
#
#                 while pos_class == neg_class:
#                     neg_class = np.random.choice(classes)
#
#                 c_p = np.where(target == pos_class)
#                 c_n = np.where(target == neg_class)
#                 if len(target == pos_class) == 2:
#                     ianc, ipos = np.random.choice(2, size=2, replace=False)
#
#                 else:
#                     ianc = np.random.randint(0, len(c_p[0]))
#                     ipos = np.random.randint(0, len(c_p[0]))
#                     while ianc == ipos:
#                         ipos = np.random.randint(0, len(c_p[0]))
#
#                 ineg = np.random.randint(0, len(c_n[0]))
#                 anc_img = data[c_p[0][ianc]]
#                 pos_img = data[c_p[0][ipos]]
#                 neg_img = data[c_n[0][ineg]]
#
#                 anc_embedding = model(anc_img.unsqueeze(0))
#                 pos_embedding = model(pos_img.unsqueeze(0))
#                 neg_embedding = model(neg_img.unsqueeze(0))
#
#                 pos_dist = torch.dist(anc_embedding, pos_embedding, p=2)
#                 neg_dist = torch.dist(anc_embedding, neg_embedding, p=2)
#
#                 if neg_dist > pos_dist:
#                     t = t + 1
#                 anchor_embeddings.append(anc_embedding.data.numpy())
#                 negative_embeddings.append(neg_embedding.data.numpy())
#                 positive_embeddings.append(pos_embedding.data.numpy())
#                 anchor_distances.append([pos_dist, neg_dist])
#                 break
#
#             total_triplets += num_triplets
#         acc = t / total_triplets
#     print('acc=', acc, 'num triplets', total_triplets)
#     for metric in metrics:
#         metric.reset()
#     epoch_time_start = time.time()
#     triplet_loss_sum = 0
#     num_triplets = 0
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # Apply network to get embeddings
#         output = model(data)
#         # Calculating loss
#         loss_outputs = loss_fn(output, target)
#         triplet_loss = loss_outputs[0]
#         num_hard_triplets = loss_outputs[1]
#         for metric in metrics:
#             metric(output, target, loss_outputs)
#         triplet_loss_sum += triplet_loss
#         num_triplets += num_hard_triplets
#         # Backward pass
#         optimizer_model.zero_grad()
#         triplet_loss.backward()
#         optimizer_model.step()
#         print('batch',batch_idx)
#     avg_triplet_loss = 0 if (num_triplets == 0) else triplet_loss_sum  # / num_triplets
#     epoch_time_end = time.time()
#     train_logs = {'loss': avg_triplet_loss}
#     # if num_triplets:
#     #     ckpter.check_on(epoch=epoch, monitor='loss', loss_acc=train_logs)
#
#     print(
#         'Epoch {}:\tAverage Triplet Loss: {:.4f}\tEpoch Time: {:.3f} hours\tNumber of valid training triplets in epoch: {}'.format(
#             epoch + 1,
#             avg_triplet_loss,
#             (epoch_time_end - epoch_time_start) / 3600,
#             num_triplets
#         )
#     )


# best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
#                                exp='batch_all').select_best(run='Run01').selected_ckpt
# model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
# model.eval()
# t = 0
# total_triplets = 0
# embeddings_data = []
# y = []
# with torch.no_grad():
#     for batch_idx, (data, target) in enumerate(test_loader):
#         if batch_idx > 1:
#             continue
#         embedding = model(data)
#         embeddings_data.append(embedding.data.numpy())
#         y.append(target.numpy())
#
# embeddings_data = np.array(embeddings_data).squeeze()
# embeddings_data = embeddings_data.reshape(-1, embeddings_data.shape[-1])
# y = np.array(y).reshape(-1)
# pca = PCA(n_components=2)
# embeddings_pca = pca.fit_transform(embeddings_data)
# cm = plt.get_cmap('gist_rainbow')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# labels = np.unique(y)
# NUM_COLORS = labels.shape[0]
# print('NUM_COLORS', NUM_COLORS)
# t = np.where(y == labels[0])
# for i in range(NUM_COLORS):
#     a = np.where(y == labels[i])
#     lines = plt.scatter(embeddings_pca[a, 0], embeddings_pca[a, 1])
#     lines.set_color(cm(i / NUM_COLORS))
#
# plt.show()

best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                               exp='batch_hard').select_best(run='Run03').selected_ckpt
model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])

model.eval()
t = 0
total_triplets = 0
anchor_embeddings = []
negative_embeddings = []
positive_embeddings = []
anchor_distances = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx > 3:
            continue
        num_triplets = len(target)
        for ind in range(num_triplets):
            classes = np.unique(target)
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)

            while len(target == pos_class) < 2:
                pos_class = np.random.choice(classes)

            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            c_p = np.where(target == pos_class)
            c_n = np.where(target == neg_class)
            if len(target == pos_class) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)

            else:
                ianc = np.random.randint(0, len(c_p[0]))
                ipos = np.random.randint(0, len(c_p[0]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(c_p[0]))

            ineg = np.random.randint(0, len(c_n[0]))
            anc_img = data[c_p[0][ianc]]
            pos_img = data[c_p[0][ipos]]
            neg_img = data[c_n[0][ineg]]

            anc_embedding = model(anc_img.unsqueeze(0))
            pos_embedding = model(pos_img.unsqueeze(0))
            neg_embedding = model(neg_img.unsqueeze(0))

            pos_dist = torch.dist(anc_embedding, pos_embedding, p=2)
            neg_dist = torch.dist(anc_embedding, neg_embedding, p=2)

            if neg_dist > pos_dist:
                t = t + 1
            anchor_embeddings.append(anc_embedding.data.numpy())
            negative_embeddings.append(neg_embedding.data.numpy())
            positive_embeddings.append(pos_embedding.data.numpy())
            anchor_distances.append([pos_dist, neg_dist])
            break

        total_triplets += num_triplets
    acc = t / total_triplets
print('acc=', acc, 'num triplets', total_triplets)
