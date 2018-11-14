import numpy as np

from Hive import Hive
from Hive.bees.BeeBinaryConstrained import BeeBinaryConstrained

from sources.problem_formulation.sensor_network_design_examples import example_1


class SensorNetworkDesignABC:

    def __init__(self,
                 sndp,
                 numb_bees=50,
                 max_itrs=200,
                 max_trials=None,
                 seed=None,
                 verbose=False,
                 theta_min=0.5,
                 theta_max=0.9):

        self.sndp = sndp

        dimensions = self.sndp.dimensions()
        bee_prototype = BeeBinaryConstrained(dimensions, self._cost_solution, self._check_constraints, theta_min,
                                             theta_max)

        self.hive = Hive.BeeHive(bee_prototype=bee_prototype,
                                 numb_bees=numb_bees,
                                 max_itrs=max_itrs,
                                 max_trials=max_trials,
                                 seed=seed,
                                 verbose=verbose)

    def run(self):
        statistics = self.hive.run()
        return statistics, self.hive.best, self.hive.calls_to_fitness

    def regenerate_seed(self, seed=None):
        self.hive.regenerate_seed(seed)

    def precisions_obtained(self, vector):
        precisions, _, _, _ = self.sndp.analyze_solution(np.array(vector))
        return precisions

    def _cost_solution(self, vector):
        return np.sum(np.array(vector) * self.sndp.costo)

    def _check_constraints(self, vector):
        _, _, _, precisions_met = self.sndp.analyze_solution(np.array(vector))
        return precisions_met.shape[0] - np.sum(precisions_met)


if __name__ == '__main__':
    hive = SensorNetworkDesignABC(example_1())
    log, best, _ = hive.run()

    # prints out best solution
    print("Fitness Value ABC: {0}".format(best.fitness))
    print("Solution ABC: {0}".format(best.vector))
