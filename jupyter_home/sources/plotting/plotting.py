import numpy as np
import matplotlib.pyplot as plt

from sources.plotting.metrics import load_historic_matrix
from sources.settings import settings_dic


def plot_historic_fitness(img_path: str, title: str, datset_name: str, settings_name: str):
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 16}

    settings = settings_dic[settings_name]
    n_generations = settings['max_itrs']

    historic_matrix = load_historic_matrix(datset_name, settings_name)

    f, axs = plt.subplots(1)
    f.set_size_inches(w=10, h=10)

    x = np.arange(0, n_generations)
    median_historic = np.median(historic_matrix, axis=0)
    min_historic = np.min(historic_matrix, axis=0)
    max_historic = np.max(historic_matrix, axis=0)

    axs.plot(x, median_historic, label="median", linewidth=3.0)
    axs.fill_between(x, min_historic, max_historic, alpha=0.5)

    axs.set_title(title, fontdict=font)
    axs.set_xlabel("generation number", fontdict=font)
    axs.set_ylabel("fitness value", fontdict=font)
    axs.grid(color='gray', linestyle='-', linewidth=0.5)

    plt.savefig(img_path)


def plot_historic_cost(img_path: str, title: str, datset_name: str, settings_name: str, max_clip: float):
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 16}

    settings = settings_dic[settings_name]
    n_generations = settings['max_itrs']

    historic_matrix = load_historic_matrix(datset_name, settings_name)

    f, axs = plt.subplots(1)
    f.set_size_inches(w=10, h=10)

    x = np.arange(0, n_generations)
    median_historic = np.median(historic_matrix, axis=0)
    min_historic = np.min(historic_matrix, axis=0)
    max_historic = np.max(historic_matrix, axis=0)

    # transform fitness to cost
    median_historic = np.clip(1/np.clip(median_historic, 1e-6, None) - 1, None, max_clip)
    min_historic = np.clip(1/np.clip(min_historic, 1e-6, None) - 1, None, max_clip)
    max_historic = np.clip(1/np.clip(max_historic, 1e-6, None) - 1, None, max_clip)

    axs.plot(x, median_historic, label="median", linewidth=3.0)
    axs.fill_between(x, min_historic, max_historic, alpha=0.5)

    axs.set_title(title, fontdict=font)
    axs.set_xlabel("generation number", fontdict=font)
    axs.set_ylabel("cost value", fontdict=font)
    axs.grid(color='gray', linestyle='-', linewidth=0.5)

    plt.savefig(img_path)





