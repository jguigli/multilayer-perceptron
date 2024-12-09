import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def plot_learning_curves(curves, name):
    plt.plot(range(len(curves)), curves)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.show()
    plt.show()

def standard_scaler(data):
    mean = np.mean(data)
    scale = np.std(data - mean)
    return (data - mean) / scale

def scaler(data):
    max_value = np.max(data)
    data_scale = (data.astype(np.float32) - max_value / 2) / max_value
    return data_scale