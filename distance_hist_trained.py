import numpy as np
from cv_importer import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import json


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- ((x - mean) / standard_deviation) ** 2)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    np.random.seed(0)
    N_chlng = 919
    d = 224  # window size 300, 224
    material = "ZnO"
    thickness_in_nm = [9483, 9563, 9690, 9819, 9925, 9945, 10089, 10137, 10285, 10385, 10485, 10968, 11006, 11056,
                       11071,
                       11093, 11477, 11621, 11623, 11675]
    N_puf = len(thickness_in_nm)
    polarization = ["hh", "hv", "vh", "vv"]
    # ind_chlng = np.random.randint(N_chlng)
    ind_ch = np.random.choice(range(N_chlng), N_chlng, replace=False)
    with open('{}/features_1.json'.format(ROOT_DIR), 'r') as fp:
        dataset = json.load(fp)
    feat = np.stack(dataset['emb']).reshape(-1, 1024)
    with open('{}/dataset.json'.format(ROOT_DIR), 'r') as fp:
        dataset = json.load(fp)
    partition = dataset['partition']
    data_x = partition['train'] + partition['validation'] + partition['test']

    score = []
    for ind_p in range(len(thickness_in_nm)):
        for ind_c in range(len(ind_ch)):
            ind_chlng = ind_ch[ind_c]
            for i in range(len(polarization)):
                for j in range(i + 1, len(polarization)):
                    try:
                        data_id = str(thickness_in_nm[ind_p]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[j]
                        features1 = feat[data_x.index(data_id)]

                        data_id = str(thickness_in_nm[ind_p]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[i]
                        features2 = feat[data_x.index(data_id)]

                        features1 = features1.flatten().astype(np.float32)
                        features2 = features2.flatten().astype(np.float32)
                        features1 = features1 / np.linalg.norm(features1)
                        features2 = features2 / np.linalg.norm(features2)
                        score.append(np.linalg.norm(features1 - features2))
                    except IndexError:
                        continue

    bin_list = np.linspace(0, 2, 50)
    # bin_list = np.linspace(0.2, 0.7, 50)
    bin_heights, bin_borders, p = plt.hist(np.array(score), density=True, bins=bin_list,
                                           label="intra-distance", edgecolor='None',
                                           alpha=0.5, color="salmon")
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
        for ind_c in range(len(polarization)):
            for i in range(len(thickness_in_nm)):
                for j in range(i + 1, len(thickness_in_nm)):
                    try:
                        data_id = str(thickness_in_nm[i]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[ind_c]
                        features1 = feat[data_x.index(data_id)]

                        data_id = str(thickness_in_nm[j]) + '_' + '{:04d}'.format(ind_chlng) + '_' + polarization[ind_c]
                        features2 = feat[data_x.index(data_id)]

                        features1 = features1.flatten().astype(np.float32)
                        features2 = features2.flatten().astype(np.float32)
                        features1 = features1 / np.linalg.norm(features1)
                        features2 = features2 / np.linalg.norm(features2)
                        score.append(np.linalg.norm(features1 - features2))
                    except IndexError:
                        continue

    bin_heights, bin_borders, p = plt.hist(np.array(score), density=True,
                                           bins=bin_list, label="inter-distance (fixed challenge)", color="cornflowerblue",
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
        for ind_c in range(len(polarization)):
            for i in range(100):
                for j in range(i + 1, 100):
                    try:
                        data_id = str(thickness_in_nm[ind_p]) + '_' + '{:04d}'.format(ind_ch[i]) + '_' + polarization[ind_c]
                        features1 = feat[data_x.index(data_id)]

                        data_id = str(thickness_in_nm[ind_p]) + '_' + '{:04d}'.format(ind_ch[j]) + '_' + polarization[ind_c]
                        features2 = feat[data_x.index(data_id)]

                        features1 = features1.flatten().astype(np.float32)
                        features2 = features2.flatten().astype(np.float32)
                        features1 = features1 / np.linalg.norm(features1)
                        features2 = features2 / np.linalg.norm(features2)
                        score.append(np.linalg.norm(features1 - features2))
                    except IndexError:
                        continue

    bin_heights, bin_borders, p = plt.hist(np.array(score),
                                           density=True, bins=bin_list, label="inter-distance (fixed puf)", color="darkseagreen",
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
    # plt.title('dist_hist_trained3')
    # plt.show()
    plt.savefig('dist_hist_trained1.pdf', bbox_inches='tight')
