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

Romain Wuilbercq, Angel Panizo-LLedot

"""

"""
This file contains the Bee proposed in the article The Artificial Bee Colony (ABC) algorithm is based on the
intelligent foraging behaviour of honey bee swarm, and was first proposed by Karaboga in 2005. Its meant to be used
in continuous unconstrained environments
"""

import copy
import random

from Hive.bees.BeePrototype import BeeProtype


class BeeContinuous(BeeProtype):
    """ This bee works in continuous unconstrained environments"""

    def dimensions(self):
        return self.dim

    def is_better(self, bee):
        return self.fitness > bee.fitness

    def probability(self, hive_population):
        """
        Computes probabilities the way Karaboga does in his classic ABC implementation
        :param hive_population: list of BeeContinuous
        :return: real value between [0,1]
        """
        values = [bee.fitness for bee in hive_population]
        max_value = max(values)
        return 0.9 * self.fitness / max_value + 0.1

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
        Generates a new Solution bee based on other bee. Uses the method described in the article:
        intelligent foraging behaviour of honey bee swarm, and was first proposed by Karaboga in 2005.

        :param bee: other bee, of the same class, to generate the solution with
        :return: a new bee that codifies the new solution generating from the input bees
        """
        zombee = self.clone()

        # draws a dimension to be crossed-over and mutated
        d = random.randint(0, zombee.dim - 1)

        # produces a mutant based on current bee and bee's friend
        zombee.vector[d] = zombee._mutate(d, bee)

        # checks boundaries
        zombee._check()

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

    def __init__(self, lower, upper, fun):
        """

        Instantiates a bee object randomly.

        Parameters:
        ----------
            :param list lower  : lower bound of solution vector
            :param list upper  : upper bound of solution vector
            :param def  fun    : evaluation function
        """
        # checks input
        assert (len(upper) == len(lower)), "'lower' and 'upper' must be a list of the same length."
        assert (fun is not None), "'fun' fitness function must not be None"

        # store bounds for checking new solutions
        self.lower = lower
        self.upper = upper
        self.fitness_fun = fun

        # store the number of dimensions of the problem
        self.dim = len(lower)

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
        Returns a new random vector selecting a random value between
        the lower and upper bounds of every dimension

        :return: a list of float
        """
        return [self.lower[i] + random.random() * (self.upper[i] - self.lower[i]) for i in range(self.dim)]

    def _check(self, dim=None):
        """
        Checks that a solution vector is contained within the
        pre-determined lower and upper bounds of the problem.
        If not, truncates the values to the upper bound and lower bound
        """
        if dim is None:
            range_ = range(self.dim)
        else:
            range_ = [dim]

        for i in range_:
            # checks lower bound
            if self.vector[i] < self.lower[i]:
                self.vector[i] = self.lower[i]

            # checks upper bound
            elif self.vector[i] > self.upper[i]:
                self.vector[i] = self.upper[i]

    def _mutate(self, dim, other_bee):
        """

        Mutates a given solution vector - i.e. for continuous
        real-values.

        Parameters:
        ----------
            :param int dim         : vector's dimension to be mutated
            :param int other_bee   : other BeeContinuos object

        """
        return self.vector[dim] + (random.random() - 0.5) * 2 * (self.vector[dim] - other_bee.vector[dim])

