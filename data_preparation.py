import numpy as np
import cv2 as cv


def crop_speckle(path, d):
    img_array = np.load(path)
    img = np.abs(img_array)

    mu = cv.moments(img)
    mc = (mu['m10'] / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))  # Get the mass centers

    x0 = int(mc[0] - d / 2)  # start point
    y0 = int(mc[1] - d / 2)
    # x1 = int(mc[0] + d / 2)  # end point
    # y1 = int(mc[1] + d / 2)
    speckle = img[y0:y0 + d, x0:x0 + d]
    return speckle


N = 200  # number of individuals
N_chlng = 919
d = 256  # window size
material = "ZnO"
thickness_in_nm = [9483, 9563, 9690, 9819, 9925, 9945, 10089, 10137, 10285, 10385, 10485, 10968, 11006, 11056, 11071,
                   11093, 11477, 11621, 11623, 11675]
N_puf = len(thickness_in_nm)
# path_folder = "../../../Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
path_folder = "/nfs/nas4/ID_IOT/ID_IOT/PUF_Data/NEW_Data/Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
date = "/2019-03-19/Run00/"
polar = ["Horizontal/hor_", "Horizontal/ver_", "Vertical/hor_", "Vertical/ver_"]
dataset = {}
# path_folder = "../../../Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
# date = "/2019-03-19/Run00/Complex Speckle/field_"
# polar = ["hor_hor_", "hor_ver_", "ver_hor_", "ver_ver_"]
features = np.zeros((N * 4, d, d))
labels = np.zeros(N * 4)
labels_puf = np.zeros(N * 4)
polariz = np.zeros(N * 4)
indices = np.random.choice(range(N_chlng * N_puf), N, replace=False)
polarization = ["hh", "hv", "vh", "vv"]
for i in range(len(indices)):
    ind_puf = int(indices[i] / N_chlng)
    ind_chlng = indices[i] - ind_puf * N_chlng
    path = path_folder + material + str(thickness_in_nm[ind_puf]) + date
    for j in range(len(polar)):
        npy_file = path + polar[j] + '{:04d}'.format(ind_chlng) + ".npy"
        speckle = crop_speckle(npy_file, d)
        # features[i * 4 + j, :] = speckle
        features[i * 4 + j, :] = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                              norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        labels[i * 4 + j] = i
        labels_puf[i * 4 + j] = ind_puf
        polariz[i*4+j] = j

# features = np.expand_dims(features, axis=3)
features = np.stack((features,)*3, axis=-1)
np.savez('dataset-puf.npz', features=features, labels=labels, polar=polariz)
print('data prepared')
