#!/usr/bin/env python

# ---- MODULE DOCSTRING
from logging import warning

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

Romain Wuilbercq,
Angel Panizo-LLedot

"""

# ---- IMPORT MODULES

import random

try:
    import numpy as np
except:
    raise ImportError("Install 'numpy' to use the library.")


class BeeHive(object):
    """

    Creates an Artificial Bee Colony (ABC) algorithm.

    The population of the hive is composed of three distinct types
    of individuals:

        1. "employees",
        2. "onlookers",
        3. "scouts".

    The employed bees and onlooker bees exploit the nectar
    sources around the hive - i.e. exploitation phase - while the
    scouts explore the solution domain - i.e. exploration phase.

    The number of nectar sources around the hive is equal to
    the number of actively employed bees and the number of employees
    is equal to the number of onlooker bees.

    """

    def regenerate_seed(self, seed=None):
        """
        Change the seed of the random number generator to use the gigen one, if None generates a new random seed.
        :param seed: an integer value
        :return: None
        """
        # generates a seed for the random number generator
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def run(self):
        """ Runs an Artificial Bee Colony (ABC) algorithm. """

        fitness_history = np.zeros((self.max_itrs, self.size))

        # print the seed
        if self.verbose:
            print("# Seed : {0}".format(self.seed))

        for itr in range(self.max_itrs):
            self.actual_itrs = itr

            # employees phase
            for index in range(self.size):
                self.send_employee(index)

            # onlookers phase
            self.send_onlookers()

            # scouts phase
            self.send_scout()

            # computes best path
            self.find_best()

            # stores convergence information
            np_fitness = np.array([bee.fitness for bee in self.population])
            fitness_history[itr, :] = np_fitness

            # prints out information about computation
            if self.verbose:
                self._verbose(itr, fitness_history)

        return fitness_history

    def __init__(self,
                 bee_prototype,
                 numb_bees    =  30   ,
                 max_itrs     = 100   ,
                 max_trials   = None  ,
                 seed         = None  ,
                 verbose      = False ):
        """

        Instantiates a bee hive object.

        1. INITIALISATION PHASE.
        -----------------------

        The initial population of bees should cover the entire search space as
        much as possible by randomizing individuals within the search
        space constrained by the prescribed lower and upper bounds.

        Parameters:
        ----------
            :param BeePrototype bee_prototype : a type of bee from the bees package to populate the hive with
            :param def numb_bees              : number of active bees within the hive
            :param int max_trials             : max number of trials without any improvment
            :param int seed                   : seed of random number generator
            :param boolean verbose            : makes computation verbose
        """

        # generates a seed for the random number generator
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # computes the number of employees
        self.size = int((numb_bees + numb_bees % 2))

        # assigns properties of algorithm
        self.dim = bee_prototype.dimensions()
        self.max_itrs = max_itrs
        self.actual_itrs = -1

        if max_trials is None:
            self.max_trials = 0.6 * self.size * self.dim
        else:
            self.max_trials = max_trials

        # initialises current best and its a solution vector
        self.best = bee_prototype
        self.solution = None
        self.calls_to_fitness = 0

        # creates a bee hive
        self.population = []
        for i in range(self.size):
            bee_i = bee_prototype.clone()
            bee_i.randomize()
            self.population.append(bee_i)

        # initializes the times the fitness function have been called
        self.calls_to_fitness = self.size

        # verbosity of computation
        self.verbose = verbose

    def find_best(self):
        """ Finds current best bee candidate. """
        for bee in self.population:
            if bee.is_better(self.best):
                self.best = bee.clone()

    def compute_probability(self):
        """

        Computes the relative chance that a given solution vector is
        chosen by an onlooker bee after the Waggle dance ceremony when
        employed bees are back within the hive.

        """
        probas = [bee.probability(self.population) for bee in self.population]
        max_probability = max(probas)

        # returns intervals of probabilities and maximum probability
        return [sum(probas[:i+1]) for i in range(self.size)], max_probability

    def send_employee(self, index):
        """

        2. SEND EMPLOYED BEES PHASE.
        ---------------------------

        During this 2nd phase, new candidate solutions are produced for
        each employed bee by cross-over and mutation of the employees.

        If the modified vector of the mutant bee solution is better than
        that of the original bee, the new vector is assigned to the bee.

        """
        # selects another bee
        bee_ix = index
        while bee_ix == index: bee_ix = random.randint(0, self.size-1)

        # produces a mutant based on current bee and bee's friend
        employed_bee = self.population[index]
        other_bee = self.population[bee_ix]
        zombee = employed_bee.crossover(other_bee, self._hive_state())

        self.calls_to_fitness += 1

        # deterministic crowding
        if zombee.is_better(employed_bee):
            self.population[index] = zombee
            self.population[index].counter = 0
        else:
            self.population[index].counter += 1

    def send_onlookers(self):
        """

        3. SEND ONLOOKERS PHASE.
        -----------------------

        We define as many onlooker bees as there are employed bees in
        the hive since onlooker bees will attempt to locally improve the
        solution path of the employed bee they have decided to follow
        after the waggle dance phase.

        If they improve it, they will communicate their findings to the bee
        they initially watched "waggle dancing".

        """

        # sends onlookers
        numb_onlookers = 0; beta = 0
        while numb_onlookers < self.size:
            # selects a new onlooker based on waggle dance
            index = self.select(beta)

            # sends new onlooker
            self.send_employee(index)

            # increments number of onlookers
            numb_onlookers += 1

    def select(self, beta):
        """

        4. WAGGLE DANCE PHASE.
        ---------------------

        During this 4th phase, onlooker bees are recruited using a roulette
        wheel selection.

        This phase represents the "waggle dance" of honey bees (i.e. figure-
        eight dance). By performing this dance, successful foragers
        (i.e. "employed" bees) can share, with other members of the
        colony, information about the direction and distance to patches of
        flowers yielding nectar and pollen, to water sources, or to new
        nest-site locations.

        During the recruitment, the bee colony is re-sampled in order to mostly
        keep, within the hive, the solution vector of employed bees that have a
        good fitness as well as a small number of bees with lower fitnesses to
        enforce diversity.

        Parameter(s):
        ------------
            :param float beta : "roulette wheel selection" parameter - i.e. 0 <= beta <= max(probas)

        """

        # computes probability intervals "online" - i.e. re-computed after each onlooker
        probas, max_probability = self.compute_probability()

        # draws a random number from U[0,1]
        phi = random.random()

        # increments roulette wheel parameter beta
        beta += phi * max_probability
        beta %= max_probability

        # selects a new potential "onlooker" bee
        for index in range(self.size):
            if beta < probas[index]:
                return index

    def send_scout(self):
        """

        5. SEND SCOUT BEE PHASE.
        -----------------------

        Identifies bees whose abandonment counts exceed preset trials limit,
        abandons it and creates a new random bee to explore new random area
        of the domain space.

        In real life, after the depletion of a food nectar source, a bee moves
        on to other food sources.

        By this means, the employed bee which cannot improve their solution
        until the abandonment counter reaches the limit of trials becomes a
        scout bee. Therefore, scout bees in ABC algorithm prevent stagnation
        of employed bee population.

        Intuitively, this method provides an easy means to overcome any local
        optima within which a bee may have been trapped.

        """

        # retrieves the number of trials for all bees
        trials = [ self.population[i].counter for i in range(self.size) ]

        # identifies the bee with the greatest number of trials
        index = trials.index(max(trials))

        # checks if its number of trials exceeds the pre-set maximum number of trials
        if trials[index] > self.max_trials:
            # creates a new scout bee randomly
            self.population[index].randomize()
            self.calls_to_fitness += 1

    def _verbose(self, itr, cost):
        """ Displays information about computation. """

        msg = "# Iter = {} | Best Fitness Value = {} | Mean Fitness Value = {} | Worse Fitness Value = {}"
        msg2 = "    Best Evaluation value so far = {}"
        print(msg.format(int(itr), np.max(cost[itr, :]), np.mean(cost[itr, :]), np.min(cost[itr, :])))
        print(msg2.format(self.best.eval_value))

    def _hive_state(self):
        """
        Return the actual state of the Hive, configuration values and actual iteration
        :return: dict with the state of the hive
        """
        state = {
            "size": self.size,
            "max_itrs": self.max_itrs,
            "actual_itrs": self.actual_itrs,
            "max_trials": self.max_trials
        }
        return state

# ---- END
