from time import perf_counter
from copy import deepcopy

from multiprocessing import Pool
from functools import partial

from sources.SensorNetworkDesignABC import SensorNetworkDesignABC
from sources.mongo_connection.mongo_connector import MongoDBConnection
from sources.mongo_connection.mongo_queries import save_iteration
from sources.mongo_connection.mongo_queries import count_iterations


def _reinitialize_db_connection():
    MongoDBConnection.reinitialize()


def _pool_worker(n_iter, datset_id: str, settings_id: str, hive):
    print("Doing iteration {0}...".format(n_iter))

    # Creates a local hive
    local_hive = deepcopy(hive)
    local_hive.regenerate_seed()

    # run the hive as normal
    init_date = perf_counter()
    historic, best, n_calls = local_hive.run()
    end_date = perf_counter()

    # store the results in the DB
    duration = str(end_date - init_date)
    cost = best.eval_value
    solution = best.vector
    violations = best.violations
    precision = local_hive.precisions_obtained(best.vector).tolist()

    save_iteration(datset_id, settings_id, cost, solution, precision, n_calls, duration, violations, historic)

    print("done!")


class ParallelExperiment:

    def __init__(self, datset_id: str, settings_id: str, num_iter: int, hive: SensorNetworkDesignABC):
        self.datset_id = datset_id
        self.settings_id = settings_id
        self.num_iter = num_iter
        self.hive = hive
        self.number_processes = num_iter

    def add_iterations_if_needed(self, processes=None):
        n_iters = count_iterations(self.datset_id, self.settings_id)
        print("{0} iterations available on DB".format(n_iters))

        if n_iters < self.num_iter:
            iterations_params = [x for x in range(self.num_iter - n_iters)]

            part_worker = partial(_pool_worker,
                                  datset_id=self.datset_id,
                                  settings_id=self.settings_id,
                                  hive=self.hive)

            with Pool(processes=processes, initializer=_reinitialize_db_connection) as pool:
                pool.map(part_worker, iterations_params)
