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