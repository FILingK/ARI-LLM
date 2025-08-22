import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def NRMSE_mean(pred, true):
    rmse = np.sqrt(np.sum((pred - true) ** 2))
    mean_true = np.sqrt(np.sum(true ** 2))
    return rmse / mean_true if mean_true != 0 else rmse


def NMAE_mean(pred, true):
    mae = np.sum(np.abs(pred - true))
    mean_true = np.sum(np.abs(true))
    return mae / mean_true if mean_true != 0 else mae



# kl
def normalize_to_probabilities(arr):
    """
    Normalize the array to a probability distribution
    """
    arr = np.asarray(arr)
    sum_arr = np.sum(arr)
    if sum_arr == 0:
        return arr  # If the sum is zero, return the array as is
    return arr / sum_arr


def KL(p, q):
    """
    Calculate Kullback-Leibler Divergence (KL Divergence) for predicted values and true values
    """
    epsilon = 1e-10
    p = np.asarray(p)
    q = np.asarray(q)

    # Normalize to probability distributions
    p = normalize_to_probabilities(p)
    q = normalize_to_probabilities(q)

    # Prevent issues from log(0)
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # Compute KL Divergence
    output = np.sum(p * np.log(p / q))

    return output


def metric(pred, true):
    nmae = NMAE_mean(pred, true)
    nrmse = NRMSE_mean(pred, true)
    kl_v = KL(true, pred)


    return nmae, nrmse, kl_v
