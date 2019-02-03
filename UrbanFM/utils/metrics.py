import pandas as pd
import numpy as np
import seaborn as sns


def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))
    
def get_MAPE(pred, real, upscale_factor=4):
    ori_real = real.copy()
    epsilon = 1 # if use small number like 1e-5 resulting in very large value
    real[real == 0] = epsilon 
    return np.mean(np.abs(ori_real - pred) / real)
