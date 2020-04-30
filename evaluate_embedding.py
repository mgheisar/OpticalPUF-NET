from utils import PairLoader, BalanceBatchSampler, Reporter
from torch.utils.data import DataLoader
from models import modelTriplet
from checkpoint import *
from metrics import *
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
best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                               exp='batch_all').select_best(run='Run01').selected_ckpt
model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
model.eval()


# #--------------------------------------------------------------------------------------
# # Clustering
# #--------------------------------------------------------------------------------------
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
# #--------------------------------------------------------------------------------------
# # All triplet
# #--------------------------------------------------------------------------------------
# batch_all = BatchAllTripletLoss(margin=0.3, squared=False, soft_margin=False)
# t = 0
# total_triplets = 0
# anchor_embeddings = []
# negative_embeddings = []
# positive_embeddings = []
# anchor_distances = []
# with torch.no_grad():
#     for batch_idx, (data, target) in enumerate(test_loader):
#         outputs = model(data)
#         batch_all_outputs = batch_all(outputs, target)
#         t += int(batch_all_outputs[1])
#         total_triplets += int(batch_all_outputs[2])
#     acc = 1 - t / total_triplets
# print('acc=', acc, 'num triplets', total_triplets)

# #--------------------------------------------------------------------------------------
# # Authentication
# #--------------------------------------------------------------------------------------
# load data X: input Y: class number
X = dataset['features']
Y = dataset['labels']

N = 100
NqueryH1 = N
n_samples = 4
# Sample data for training and test

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

data_x = np.swapaxes(data_x, 1, 3)  # N,1,d,d
H1_x = np.swapaxes(H1_x, 1, 3)
H0_x = np.swapaxes(H0_x, 1, 3)

data_x = torch.tensor(data_x, dtype=torch.float)
H1_x = torch.tensor(H1_x, dtype=torch.float)
H0_x = torch.tensor(H0_x, dtype=torch.float)

model = modelTriplet(embedding_dimension=128, pretrained=False)
best_model_filename = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                               exp='batch_hard').select_best(run='Run03').selected_ckpt
model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
model.eval()
with torch.no_grad():
    embedding_data = model(data_x).data.numpy()
    embedding_H1 = model(H1_x).data.numpy()
    embedding_H0 = model(H0_x).data.numpy()

H0_claimed_id = np.random.randint(0, N, size=NqueryH0).astype(np.int)

D00 = np.linalg.norm(embedding_H0 - embedding_data[H0_claimed_id, :], axis=1)

xsorted = np.argsort(data_id.squeeze())
ypos = np.searchsorted(data_id.squeeze()[xsorted], H1_id.squeeze())
temp = xsorted[ypos]
D11 = np.linalg.norm(embedding_H1 - embedding_data[temp.astype(np.int), :], axis=1)

D0 = np.sort(D00)
D1 = np.sort(D11)

Pfp = 0.05
tau = D0[int(Pfp*NqueryH0)]
Pfn = 1 - np.count_nonzero(D1 <= tau) / NqueryH1
print('pfn', Pfn)