import numpy as np
from cv_importer import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- ((x - mean) / standard_deviation) ** 2)


if __name__ == '__main__':
    np.random.seed(0)
    N_chlng = 919
    d = 300  # window size 300, 224
    material = "ZnO"
    thickness_in_nm = [9483, 9563, 9690, 9819, 9925, 9945, 10089, 10137, 10285, 10385, 10485, 10968, 11006, 11056,
                       11071,
                       11093, 11477, 11621, 11623, 11675]
    N_puf = len(thickness_in_nm)
    # path_folder = "/nfs/nas4/ID_IOT/ID_IOT/PUF_Data/NEW_Data/Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
    # date = "/2019-03-19/Run00/"
    # polar = ["Horizontal/hor_", "Horizontal/ver_", "Vertical/hor_", "Vertical/ver_"]
    path_folder = "/nfs/nas4/ID_IOT/ID_IOT/PUF_Data/NEW_Data/Pritam TM Data1/New setup/NA_0.95/deltaV_0.03/"
    date = "/2019-03-19/Run00/Complex Speckle/field_"
    polar = ["hor_hor_", "hor_ver_", "ver_hor_", "ver_ver_"]
    polarization = ["hh", "hv", "vh", "vv"]
    # ind_chlng = np.random.randint(N_chlng)
    ind_ch = np.random.choice(range(N_chlng), N_chlng, replace=False)
    score = []
    for ind_p in range(len(thickness_in_nm)):
        pathh = path_folder + material + str(thickness_in_nm[ind_p]) + date
        for ind_c in range(len(ind_ch)):
            ind_chlng = ind_ch[ind_c]
            for i in range(len(polar)):
                for j in range(i + 1, len(polar)):
                    npy_file = pathh + polar[j] + '{:04d}'.format(ind_chlng) + ".npy"
                    # data_id = str(thickness_in_nm[i]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[j]
                    # print(data_id)
                    img_array = np.abs(np.load(npy_file))
                    # plt.imshow(img_array)
                    # plt.show()
                    # speckle = crop_speckle_c1(npy_file, d)
                    speckle = cv.resize(img_array, (d, d))
                    # features1 = speckle
                    features1 = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                             norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

                    npy_file = pathh + polar[i] + '{:04d}'.format(ind_chlng) + ".npy"
                    img_array = np.abs(np.load(npy_file))
                    speckle = cv.resize(img_array, (d, d))
                    # features2 = speckle
                    features2 = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                             norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                    features1 = features1.flatten().astype(np.float32)
                    features2 = features2.flatten().astype(np.float32)
                    features1 = features1 / np.linalg.norm(features1)
                    features2 = features2 / np.linalg.norm(features2)
                    score.append(np.linalg.norm(features1 - features2))

    bin_list = np.linspace(0.5, 1, 50)
    # bin_list = np.linspace(0.2, 0.7, 50)
    bin_heights, bin_borders, p = plt.hist(np.array(score), density=True, label="intra-distance",
                                           bins=bin_list, edgecolor='None', alpha=0.5, color="salmon")
    for item in p:
        item.set_height(item.get_height() / np.sum(bin_heights))
    bin_heights = bin_heights / np.sum(bin_heights)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color="red", alpha=0.5)

    score = []
    for ind_p in range(len(ind_ch)):
        ind_chlng = ind_ch[ind_p]
        for ind_c in range(len(polar)):
            for i in range(len(thickness_in_nm)):
                for j in range(i + 1, len(thickness_in_nm)):
                    pathh = path_folder + material + str(thickness_in_nm[i]) + date
                    npy_file = pathh + polar[ind_c] + '{:04d}'.format(ind_chlng) + ".npy"
                    # data_id = str(thickness_in_nm[i]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[j]
                    # print(data_id)
                    img_array = np.abs(np.load(npy_file))
                    # plt.imshow(img_array)
                    # plt.show()
                    # speckle = crop_speckle_c1(npy_file, d)
                    speckle = cv.resize(img_array, (d, d))
                    # features1 = speckle
                    features1 = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                             norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

                    pathh = path_folder + material + str(thickness_in_nm[j]) + date
                    npy_file = pathh + polar[ind_c] + '{:04d}'.format(ind_chlng) + ".npy"
                    img_array = np.abs(np.load(npy_file))
                    speckle = cv.resize(img_array, (d, d))
                    # features2 = speckle
                    features2 = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                             norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

                    features1 = features1.flatten().astype(np.float32)
                    features2 = features2.flatten().astype(np.float32)
                    features1 = features1 / np.linalg.norm(features1)
                    features2 = features2 / np.linalg.norm(features2)
                    score.append(np.linalg.norm(features1 - features2))

    bin_heights, bin_borders, p = plt.hist(np.array(score), density=True, bins=bin_list,
                                           label="inter-distance (fixed challenge)", color="cornflowerblue",
                                           edgecolor='None', alpha=0.5)
    for item in p:
        item.set_height(item.get_height() / np.sum(bin_heights))

    bin_heights = bin_heights / np.sum(bin_heights)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color="mediumblue", alpha=0.5)

    score = []
    for ind_p in range(len(thickness_in_nm)):
        pathh = path_folder + material + str(thickness_in_nm[ind_p]) + date
        for ind_c in range(len(polar)):
            for i in range(100):
                for j in range(i + 1, 100):
                    ind_chlng = ind_ch[i]
                    npy_file = pathh + polar[ind_c] + '{:04d}'.format(ind_chlng) + ".npy"
                    # data_id = str(thickness_in_nm[i]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[j]
                    # print(data_id)
                    img_array = np.abs(np.load(npy_file))
                    # plt.imshow(img_array)
                    # plt.show()
                    # speckle = crop_speckle_c1(npy_file, d)
                    speckle = cv.resize(img_array, (d, d))
                    # features1 = speckle
                    features1 = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                             norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

                    ind_chlng = ind_ch[j]
                    npy_file = pathh + polar[ind_c] + '{:04d}'.format(ind_chlng) + ".npy"
                    img_array = np.abs(np.load(npy_file))
                    speckle = cv.resize(img_array, (d, d))
                    # features2 = speckle
                    features2 = cv.normalize(src=speckle, dst=None, alpha=0, beta=255,
                                             norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                    features1 = features1.flatten().astype(np.float32)
                    features2 = features2.flatten().astype(np.float32)
                    features1 = features1 / np.linalg.norm(features1)
                    features2 = features2 / np.linalg.norm(features2)
                    score.append(np.linalg.norm(features1 - features2))

    bin_heights, bin_borders, p = plt.hist(np.array(score), bins=bin_list,
                                           density=True, label="inter-distance (fixed puf)", color="darkseagreen",
                                           edgecolor='None', alpha=0.5)
    for item in p:
        item.set_height(item.get_height() / np.sum(bin_heights))

    bin_heights = bin_heights / np.sum(bin_heights)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color="lime", alpha=0.5)

    plt.ylim(0, 0.7)
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig('dist_hist_complex.pdf', bbox_inches='tight')
