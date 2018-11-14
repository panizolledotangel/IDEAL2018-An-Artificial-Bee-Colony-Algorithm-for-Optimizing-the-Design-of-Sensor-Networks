import pickle
from typing import Tuple

import numpy as np

from sources.mongo_connection.mongo_queries import count_iterations, find_iteration
from sources.settings import settings_dic


def load_historic_matrix(datset_name: str, settings_name: str) -> np.array:
    itrs_available = count_iterations(datset_name, settings_name)

    settings = settings_dic[settings_name]
    n_generations = settings['max_itrs']

    historic_matrix = np.zeros((itrs_available, n_generations))
    with find_iteration(datset_name, settings_name) as cursor:
        for n_itr, actual_itr in enumerate(cursor):
            itr_historic = pickle.loads(actual_itr['historic'])
            historic_matrix[n_itr, :] = np.max(itr_historic, axis=1)

    return historic_matrix


def max_mean_std(datset_name: str, settings_name: str) -> Tuple[float, float, float]:
    itrs_available = count_iterations(datset_name, settings_name)

    cost_vector = np.zeros(itrs_available)
    with find_iteration(datset_name, settings_name) as cursor:
        for n_itr, actual_itr in enumerate(cursor):
            cost_vector[n_itr] = actual_itr['cost']

    return float(np.min(cost_vector)), float(np.mean(cost_vector)), float(np.std(cost_vector))
