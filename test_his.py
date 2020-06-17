from history import History
import dill
import numpy as np
import matplotlib.pyplot as plt

run_name = "Run01-hardv2"
train_hist = History(name='train_hist' + run_name)
validation_hist = History(name='validation_hist' + run_name)
triplet_method = "batch_hardv2"
train_hist = dill.load(open("ckpt/" + triplet_method + train_hist.name + ".pickle", "rb"))
validation_hist = dill.load(open("ckpt/" + triplet_method + validation_hist.name + ".pickle", "rb"))
loss, acc, acc001 = np.zeros(len(train_hist.loss)), np.zeros(len(train_hist.loss)), np.zeros(len(train_hist.loss))
nonzeros, loss_all_avg = np.zeros(len(train_hist.loss)), np.zeros(len(train_hist.loss))
for i in range(len(train_hist.loss)):
    loss[i] = train_hist.loss[i].cpu()
    acc[i] = train_hist.acc[i]
    acc001[i] = train_hist.acc001[i]
    nonzeros[i] = train_hist.nonzerostriplets[i]
    loss_all_avg[i] = train_hist.loss_all_avg[i]
np.savez('train_hist.npz', loss=loss, acc=acc, acc001=acc001, nonzeros=nonzeros, loss_all_avg=loss_all_avg)

loss, acc, acc001 = np.zeros(len(validation_hist.loss)), np.zeros(len(validation_hist.loss)), np.zeros(len(validation_hist.loss))
nonzeros, loss_all_avg = np.zeros(len(validation_hist.loss)), np.zeros(len(validation_hist.loss))
for i in range(len(validation_hist.loss)):
    loss[i] = validation_hist.loss[i].cpu()
    acc[i] = validation_hist.acc[i]
    acc001[i] = validation_hist.acc001[i]
    nonzeros[i] = validation_hist.nonzerostriplets[i]
    loss_all_avg[i] = validation_hist.loss_all_avg[i]
np.savez('validation_hist.npz', loss=loss, acc=acc, acc001=acc001, nonzeros=nonzeros, loss_all_avg=loss_all_avg)

np.load('validation_hist.npz')
np.load('train_hist.npz')
