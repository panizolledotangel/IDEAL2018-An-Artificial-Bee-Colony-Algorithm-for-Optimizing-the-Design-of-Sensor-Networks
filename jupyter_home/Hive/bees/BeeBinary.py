#!/usr/bin/env python

# ---- MODULE DOCSTRING
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

Author:
------

Angel Panizo-LLedot

"""

"""
This file contains the Bee proposed in the article 'Kashan, M. H., Nahavandi, N., & Kashan, A. H. (2012). DisABC: a new 
artificial bee colony algorithm for binary optimization. Applied Soft Computing, 12(1), 342-352.'. Its meant to be used
in binary unconstrained environments
"""

import copy
import random

import numpy as np

from Hive.bees.BeePrototype import BeeProtype


class BeeBinary(BeeProtype):
    """ This bee works in binary unconstrained environments"""

    @classmethod
    def _branch_and_bound(cls, theta, similarity, n0, n1):
        """
        Branch and bound, depth first method that minimizes the next constraint problem

        min | 1 - m11/(m11 + m10 + m01) - (theta * (1-similarity)) |
        c1: m11 + m01 == n1
        c2: m10 <= n0

        :param theta: float constant
        :param similarity: float constant
        :param n0: int constant
        :param n1: int constant
        :return: m11, m10
        """

        # constants
        A = theta * (1 - similarity)

        # variables bounds
        m11_ub = n1
        m10_ub = n0
        m01_ub = n1

        # solution [m11, m10, m01]
        best_solution = [n1, n0, n1]
        best_score = float('inf')

        for m11 in range(0, m11_ub+1):
            for m01 in range(0, m01_ub+1):
                if m11 + m01 == n1:
                    for m10 in range(0, m10_ub):
                        if m11 + m01 + m10 > 0:
                            score = abs(1.0 - m11 / (m11 + m10 + m01) - A)
                            if score < best_score:
                                best_score = score
                                best_solution = [m11, m10, m01]

        return best_solution[0], best_solution[1]

    def dimensions(self):
        return self.dim

    def is_better(self, bee):
        return self.fitness > bee.fitness

    def probability(self, hive_population):
        """
        Computes probabilities the way described in the article:
        Kashan, M. H., Nahavandi, N., & Kashan, A. H. (2012). DisABC: a new
        artificial bee colony algorithm for binary optimization. Applied Soft Computing, 12(1), 342-352.

        :param hive_population: list of BeeContinuous
        :return: real value between [0,1]
        """
        values = [bee.fitness for bee in hive_population]
        sum_values = sum(values)

        return self.fitness / sum_values

    def randomize(self):
        """
        Transforms this bee's solution into a new random one, selecting a value between l
        ower and upper bound for each dimension
        :return: None
        """
        # creates a random solution vector
        self.vector = self._random()

        # computes fitness of solution vector
        value = self.fitness_fun(self.vector)
        self.fitness = self._fitness(value)
        self.eval_value = value

        # initialises trial limit counter - i.e. abandonment counter
        self.counter = 0

    def crossover(self, bee, hive_state):
        """
        Generates a new Solution bee based on other bee. Uses the method described  in the section 3.2 of the article:
        Kashan, M. H., Nahavandi, N., & Kashan, A. H. (2012). DisABC: a new
        artificial bee colony algorithm for binary optimization. Applied Soft Computing, 12(1), 342-352.

        :param bee: other bee, of the same class, to generate the solution with
        :return: a new bee that codifies the new solution generating from the input bees
        """
        zombee = self.clone()

        # calculate new theta
        theta = self.theta_max - hive_state['actual_itrs']*((self.theta_max - self.theta_min)/hive_state['max_itrs'])

        # calculate similarity
        similarity = zombee._jaccard_coeficient(bee)

        # calculate the number zeros and ones in this bee
        vector = np.array(zombee.vector)
        n0 = np.where(vector == 0)[0].shape[0]
        n1 = np.where(vector == 1)[0].shape[0]

        # solve integer programming to find number of bits to change
        m11, m10 = BeeBinary._branch_and_bound(theta, similarity, n0, n1)

        # produces a mutant based on number of bits found by the solver
        zombee.vector = zombee._mutate(m11, m10)

        # computes fitness of mutant
        value = zombee.fitness_fun(zombee.vector)
        zombee.fitness = zombee._fitness(value)
        zombee.eval_value = value

        return zombee

    def clone(self):
        """
        Makes an exact copy of this bee
        :return: a new Bee equals to this one
        """
        return copy.deepcopy(self)

    def __init__(self, dimensions, fun, theta_min=0.5, theta_max=0.9):
        """

        Instantiates a bee object randomly.

        Parameters:
        ----------
            :param int dimensions  : number of dimensions the solution must have
            :param def  fun    : evaluation function
            :param float theta_min : governs the minimum radius around a solutions being exploited
            :param float theta_max : governs the maximum radius around a solutions being exploited
        """
        # checks input
        assert theta_min <= theta_max, "theta_min must be smaller or equal than theta_max"
        assert (fun is not None), "'fun' fitness function must not be None"

        self.theta_min = theta_min
        self.theta_max = theta_max

        # store fitness function
        self.fitness_fun = fun

        # store the number of dimensions of the problem
        self.dim = dimensions

        # creates a random solution vector
        self.vector = self._random()

        # dummy fitness value
        self.fitness = 0.0
        self.eval_value = float('nan')

        # initialises trial limit counter - i.e. abandonment counter
        self.counter = 0

    def _fitness(self, value):
        """

        Evaluates the fitness of a solution vector.

        The fitness is a measure of the quality of a solution.
        :param value: value returned by the fitness function
        :returns: the fitness value of this bee
        """

        if value >= 0:
            fitness = 1 / (1 + value)
        else:
            fitness = 1 + abs(value)
        return fitness

    def _random(self):
        """
        Returns a new random vector following a Bernoulli distribution

        :return: a list of 0,1
        """
        bin_vect = [0]*self.dim
        for i in range(self.dim):
            if random.random() >= 0.5:
                bin_vect[i] = 1

        return bin_vect

    def _mutate(self, m11, m10):
        """

        Mutates a given solution vector following method described in Algorithm NBSG of article:
        Kashan, M. H., Nahavandi, N., & Kashan, A. H. (2012). DisABC: a new
        artificial bee colony algorithm for binary optimization. Applied Soft Computing, 12(1), 342-352.

        Parameters:
        ----------
            :param int m11 : number of bits of the solution that are 1 in the old solution and 1 in the new solution
            :param int m10 : number of bits of the solution that are 0 in the old solution and 1 in the new solution

        """
        solution = np.zeros(self.dim)

        vector = np.array(self.vector)

        if m11 > 0:
            index_11 = np.random.choice(np.where(vector == 1)[0], size=m11, replace=False)
            solution[index_11] = 1

        if m10 > 0:
            index_10 = np.random.choice(np.where(vector == 0)[0], size=m10, replace=False)
            solution[index_10] = 1

        return solution.tolist()

    def _jaccard_coeficient(self, other_bee):
        """
        Calculates the Jaccard Coeficient of similarity between this bee and the given bee
        :param other_bee: BeeBinary to calculate the dissimilarity with
        :return: a real value between [0,1]
        """
        bee_i = np.array(self.vector)
        bee_j = np.array(other_bee.vector)

        m11 = np.intersect1d(np.where(bee_i == 1)[0], np.where(bee_j == 1)[0]).shape[0]
        m10 = np.intersect1d(np.where(bee_i == 1)[0], np.where(bee_j == 0)[0]).shape[0]
        m01 = np.intersect1d(np.where(bee_i == 0)[0], np.where(bee_j == 1)[0]).shape[0]

        return m11/(m11 + m10 + m01)



