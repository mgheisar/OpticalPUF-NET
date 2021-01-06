import numpy as np
from cv_importer import *
import json
from os import path
import skimage.segmentation as ski
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as patches


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- ((x - mean) / standard_deviation) ** 2)


def crop_speckle_c1(pathh, d):
    img_array = np.load(pathh)
    img = np.abs(img_array)
    img = cv.normalize(src=img, dst=None, alpha=0, beta=255,
                       norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    mu = cv.moments(img)
    mc = (mu['m10'] / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))  # Get the mass centers

    x0 = int(mc[0] - d / 2)  # start point
    y0 = int(mc[1] - d / 2)
    x1 = int(mc[0] + d / 2)  # end point
    y1 = int(mc[1] + d / 2)
    speckle_img = img[y0:y0 + d, x0:x0 + d]

    # plt.imshow(img)
    # rect = patches.Rectangle((x0, y0), d, d, linewidth=2, edgecolor='r', facecolor='none')
    # plt.add_patch(rect)  # Add the patch to the Axes
    # plt.show()

    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((x0, y0), d, d, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)  # Add the patch to the Axes
    # plt.show()
    plt.savefig('speckle1_complex_c1.pdf', bbox_inches='tight')
    return speckle_img


def crop_speckle_c2(pathh, d):
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

    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((x0, y0), d, d, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)  # Add the patch to the Axes
    plt.savefig('speckle1_complex_c2.pdf', bbox_inches='tight')
    # plt.show()
    return speckle_img


train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1
N_chlng = 919
d = 300  # window size 300, 224
material = "ZnO"
thickness_in_nm = [9483, 9563, 9690, 9819, 9925, 9945, 10089, 10137, 10285, 10385, 10485, 10968, 11006, 11056, 11071,
                   11093, 11477, 11621, 11623, 11675]
N_puf = len(thickness_in_nm)
N = N_chlng * N_puf  # number of individuals
# path_folder = "/nfs/nas4/ID_IOT/ID_IOT/PUF_Data/NEW_Data/Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
# date = "/2019-03-19/Run00/"
# polar = ["Horizontal/hor_", "Horizontal/ver_", "Vertical/hor_", "Vertical/ver_"]

path_folder = "/nfs/nas4/ID_IOT/ID_IOT/PUF_Data/NEW_Data/Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
date = "/2019-03-19/Run00/Complex Speckle/field_"
polar = ["hor_hor_", "hor_ver_", "ver_hor_", "ver_ver_"]
features = np.zeros((1, d, d))
indices = np.random.choice(range(N_chlng * N_puf), N, replace=False)
polarization = ["hh", "hv", "vh", "vv"]
ind_puf = 0
ind_c = [360, 890, 917]
j = 0
ind_chlng = 917
pathh = path_folder + material + str(thickness_in_nm[ind_puf]) + date

npy_file = pathh + polar[j] + '{:04d}'.format(ind_chlng) + ".npy"
data_id = str(thickness_in_nm[ind_puf]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[j]
print(data_id)
img_array = crop_speckle_c2(npy_file, d)
# img_array = np.load(npy_file)
# img_array = np.abs(img_array)
# img_array = cv.normalize(src=img_array, dst=None, alpha=0, beta=255,
#                          norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# img_array = cv.resize(img_array, (d, d))
# plt.imshow(img_array)
# plt.savefig('speckle1_resized.pdf', bbox_inches='tight')


# bin_list = np.linspace(0, 0.01, 10)
# vec_norm = img_array.flatten() / np.linalg.norm(img_array.flatten())
# bin_heights, bin_borders, p = plt.hist(vec_norm, density=True, bins=bin_list)
# for item in p:
#     item.set_height(item.get_height() / np.sum(bin_heights))

# bin_heights = bin_heights / np.sum(bin_heights)
# bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
# popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
# x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
# plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt))
# plt.ylim(0, 1)
# plt.show()
