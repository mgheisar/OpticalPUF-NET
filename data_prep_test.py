import numpy as np
from cv_importer import *
import json
from os import path
import skimage.segmentation as ski


def crop_speckle1(pathh, d):
    img_array = np.load(pathh)
    img = np.abs(img_array)

    mu = cv.moments(img)
    mc = (mu['m10'] / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))  # Get the mass centers

    x0 = int(mc[0] - d / 2)  # start point
    y0 = int(mc[1] - d / 2)
    # x1 = int(mc[0] + d / 2)  # end point
    # y1 = int(mc[1] + d / 2)
    speckle_img = img[y0:y0 + d, x0:x0 + d]
    return speckle_img


def crop_speckle(pathh, d):
    img_array = np.load(pathh)
    img = np.abs(img_array)
    img = cv.normalize(src=img, dst=None, alpha=0, beta=255,
                       norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    img1 = ski.clear_border(img > 200.0)
    y, x = np.nonzero(img1)
    c_x = x.mean()
    c_y = y.mean()
    if c_x < int(d / 2):
        c_x = int(d / 2)
    elif c_x >= (len(img) - int(d / 2)):
        c_x = len(img) - int(d / 2) - 1
    if c_y < int(d / 2):
        c_y = int(d / 2)
    elif c_y >= (len(img) - int(d / 2)):
        c_y = len(img) - int(d / 2) - 1
    x0 = int(c_x - int(d / 2))  # start point
    y0 = int(c_y - int(d / 2))
    speckle_img = img[y0:y0 + d, x0:x0 + d]
    return speckle_img


#  -------- Save the matrix for data to use pairloader (utils_test)
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1
N_chlng = 919
d = 224  # window size 256
material = "ZnO"
thickness_in_nm = [9483, 9563, 9690, 9819, 9925, 9945, 10089, 10137, 10285, 10385, 10485, 10968, 11006, 11056, 11071,
                   11093, 11477, 11621, 11623, 11675]
N_puf = len(thickness_in_nm)
N = N_chlng * N_puf  # number of individuals
# path_folder = "../../../Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
path_folder = "/nfs/nas4/ID_IOT/ID_IOT/PUF_Data/NEW_Data/Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
date = "/2019-03-19/Run00/"
polar = ["Horizontal/hor_", "Horizontal/ver_", "Vertical/hor_", "Vertical/ver_"]
partition = {'train': [], 'validation': [], 'test': []}
labels_train = {}
labels_test = {}
labels_validation = {}
# path_folder = "../../../Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
# date = "/2019-03-19/Run00/Complex Speckle/field_"
# polar = ["hor_hor_", "hor_ver_", "ver_hor_", "ver_ver_"]
features = np.zeros((1, d, d))
indices = np.random.choice(range(N_chlng * N_puf), N, replace=False)
polarization = ["hh", "hv", "vh", "vv"]
n_train = int(len(indices) * train_ratio)
n_validation = int(len(indices) * validation_ratio)
n_test = len(indices) - (n_train + n_validation)
X_train = np.zeros((n_train * 4, 3, d, d), dtype=np.uint8)
X_validation = np.zeros((n_validation * 4, 3, d, d), dtype=np.uint8)
X_test = np.zeros((n_test * 4, 3, d, d), dtype=np.uint8)
labels_train = np.zeros(n_train * 4)
labels_validation = np.zeros(n_validation * 4)
labels_test = np.zeros(n_test * 4)
t1, t2, t3 = -1, -1, -1
for i in range(len(indices)):
    ind_puf = int(indices[i] / N_chlng)
    ind_chlng = indices[i] - ind_puf * N_chlng
    pathh = path_folder + material + str(thickness_in_nm[ind_puf]) + date
    for j in range(len(polar)):
        npy_file = pathh + polar[j] + '{:04d}'.format(ind_chlng) + ".npy"
        data_id = str(thickness_in_nm[ind_puf]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[j]
        if not path.exists('data/' + data_id + '.npy'):
            speckle = crop_speckle(npy_file, d)
            # features[i * 4 + j, :] = speckle
            features = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                    norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            features = np.stack((features,) * 3, axis=-1)
            features = np.swapaxes(features, 0, 2)  # 3,d,d
            # features = np.expand_dims(features, axis=0)
            np.save('data/' + data_id + '.npy', features)
        if i < n_train:
            t1 += 1
            X_train[t1, :] = np.load('data/' + data_id + '.npy')
            labels_train[t1] = i
        elif i < n_train + n_validation:
            t2 += 1
            X_validation[t2, :] = np.load('data/' + data_id + '.npy')
            labels_validation[t2] = i
        else:
            t3 += 1
            X_test[t3, :] = np.load('data/' + data_id + '.npy')
            labels_test[t3] = i
        # labels[data_id] = i
labels = {'train': labels_train, 'validation': labels_validation, 'test': labels_test}
partition = {'train': X_train, 'validation': X_validation, 'test': X_test}

np.savez('labels.npz', train=labels_train, validation=labels_validation, test=labels_test)
np.savez('partition.npz', train=X_train, validation=X_validation, test=X_test)
# dataset = {'partition': partition, 'labels': labels}
# with open('dataset-puf-all.json', 'w') as f_out:
#     json.dump(labels, f_out)
print('data prepared')
