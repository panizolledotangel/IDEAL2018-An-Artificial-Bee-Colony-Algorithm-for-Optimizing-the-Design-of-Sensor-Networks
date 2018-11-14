# ---- MODULE DOCSTRING
import math

__doc__ = """

(C) Hive, Romain Wuilbercq, 2017
     _
    /_/_      .'''.
 =O(_)))) ...'     `.
    \_\              `.    .'''X
                       `..'
.---.  .---..-./`) ,---.  ,---.   .-''-.
|   |  |_ _|\ .-.')|   /  |   | .'_ _   \
|   |  ( ' )/ `-' \|  |   |  .'/ ( ` )   '
|   '-(_{;}_)`-'`"`|  | _ |  |. (_ o _)  |
|      (_,_) .---. |  _( )_  ||  (_,_)___|
| _ _--.   | |   | \ (_ o._) /'  \   .---.
|( ' ) |   | |   |  \ (_,_) /  \  `-'    /
(_{;}_)|   | |   |   \     /    \       /
'(_,_) '---' '---'    `---`      `'-..-'

The Artificial Bee Colony (ABC) algorithm is based on the
intelligent foraging behaviour of honey bee swarm, and was first proposed
by Karaboga in 2005.

"""
import unittest

import numpy as np

# from Hive import Utilities
from Hive import Hive
from Hive.bees.BeeContinuous import BeeContinuous
from Hive.bees.BeeContinuousConstrained import BeeContinuousConstrained
from Hive.bees.BeeBinary import BeeBinary
from Hive.bees.BeeBinaryConstrained import BeeBinaryConstrained


class TestHive(unittest.TestCase):

    @classmethod
    def continuous_evaluator(cls, vector, a=1, b=100):
        """

        The Rosenbrock function is a non-convex function used as a performance test
        problem for optimization algorithms introduced by Howard H. Rosenbrock in
        1960. It is also known as Rosenbrock's valley or Rosenbrock's banana
        function.

        The function is defined by

                            f(x, y) = (a-x)^2 + b(y-x^2)^2

        It has a global minimum at (x, y) = (a, a**2), where f(x, y) = 0.

        """

        vector = np.array(vector)
        return (a - vector[0]) ** 2 + b * (vector[1] - vector[0] ** 2) ** 2

    @classmethod
    def continuous_constrained_evaluator(cls, vector):
        """
        minimize:

        f(x) = x**2 + y**2

        :param vector: solution vector
        :return: float values of the function, return inverse for minimization
        """
        vector = np.array(vector)
        return np.sum(vector**2)

    @classmethod
    def continuous_constrained_check(cls, vector):
        """
        Check the constraints:

        g1(x): x + y - 2 >= 0

        :param vector: solution vector
        :return: int constraints met
        """
        violations = 0
        vector = np.array(vector)

        # constraint 1
        if sum(vector) - 2.0 < 0:
            violations += 1

        return violations

    @classmethod
    def binary_evaluator(cls, vector):
        """
        MAXCUT problem for the graph composed of the next edge list:
        [ (0, 1),
          (0, 2),
          (0, 3),
          (1, 2),
          (1, 3),
          (2, 3),
          (2, 4),
          (2, 5),
          (4, 5) ]

        :param vector: binary vector, position i is 0 if node i is on group 0, 1 if on group 1
        :return: number of edges that its extreme nodes are in opposite groups
        """

        edge_list = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (2, 5),
            (4, 5)
        ]

        n_cuts = 0
        for n_i, n_j in edge_list:
            if vector[n_i] != vector[n_j]:
                n_cuts += 1

        if n_cuts == 0:
            return float('inf')
        else:
            return 1.0/n_cuts

    @classmethod
    def binary_constrained_evaluator(cls, vector):
        """
        Knapsack problem

        # P is total weight capacity of sack
        # weights and utilities are also specified
        P = 165
        weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
        utilities = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])

        :param vector: binary vector, position i is 0 if node i is on group 0, 1 if on group 1
        :return: total utility of the sack
        """
        utilities = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
        return 1/np.sum(vector*utilities)

    @classmethod
    def binary_constrains_check(cls, vector):
        """
        Knapsack problem

        # P is total weight capacity of sack
        # weights and utilities are also specified
        P = 165
        weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
        utilities = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])

        :param vector: binary vector, position i is 0 if node i is on group 0, 1 if on group 1
        :return: 1 if the sack is overweight, 0 other wise
        """
        P = 165
        weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])

        violations = 0
        if np.sum(vector*weights) > P:
            violations += 1

        return violations

    def test_standard(self):
        # creates model
        ndim = int(2)
        bee_prototype = BeeContinuous(lower=[0] * ndim,
                                      upper=[10] * ndim,
                                      fun=TestHive.continuous_evaluator)

        model = Hive.BeeHive(bee_prototype=bee_prototype,
                             numb_bees=100,
                             max_itrs=2000)

        # runs model
        model.run()

        # prints out best solution
        print("Fitness Value ABC: {0}".format(model.best.fitness))
        print("Solution ABC: {0}".format(model.best.vector))
        assert model.best is not None

    def test_continuous_constrained(self):
        # creates model
        ndim = int(2)

        bee_prototype = BeeContinuousConstrained(lower=[0]*ndim,
                                                 upper=[10]*ndim,
                                                 fun=TestHive.continuous_constrained_evaluator,
                                                 check_constraints=TestHive.continuous_constrained_check)

        model = Hive.BeeHive(bee_prototype=bee_prototype,
                             numb_bees=50,
                             max_itrs=200)

        # runs model
        model.run()

        # prints out best solution
        print("Fitness Value ABC: {0}".format(model.best.fitness))
        print("Solution ABC: {0}".format(model.best.vector))
        assert model.best is not None

    def test_binary(self):
        # creates model
        ndim = int(6)

        bee_prototype = BeeBinary(dimensions=ndim,
                                  fun=TestHive.binary_evaluator)

        model = Hive.BeeHive(bee_prototype=bee_prototype,
                             numb_bees=50,
                             max_itrs=200)

        # runs model
        model.run()

        # prints out best solution
        print("Fitness Value ABC: {0}".format(model.best.fitness))
        print("Solution ABC: {0}".format(model.best.vector))
        assert model.best is not None

    def test_binary_constrained(self):
        # creates model
        ndim = int(10)

        bee_prototype = BeeBinaryConstrained(dimensions=ndim,
                                             fun=TestHive.binary_constrained_evaluator,
                                             check_constraints_fun=TestHive.binary_constrains_check)

        model = Hive.BeeHive(bee_prototype=bee_prototype,
                             numb_bees=50,
                             max_itrs=200)

        # runs model
        model.run()

        # prints out best solution
        print("Fitness Value ABC: {0}".format(model.best.fitness))
        print("Solution ABC: {0}".format(model.best.vector))
        assert model.best is not None


if __name__ == '__main__':
    unittest.main()
