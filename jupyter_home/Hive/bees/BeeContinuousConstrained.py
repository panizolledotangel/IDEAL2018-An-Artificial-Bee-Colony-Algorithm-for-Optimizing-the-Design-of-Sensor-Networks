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
This file contains the Bee proposed in the article "Karaboga, D., & Akay, B. (2011). A modified artificial bee colony 
(ABC) algorithm for constrained optimization problems. Applied soft computing, 11(3), 3021-3031". Its meant to be used
in continuous constrained environments
"""

import random
import sys

from Hive.bees.BeeContinuous import BeeContinuous


class BeeContinuousConstrained(BeeContinuous):
    """
    This bee works in continuous constrained environments, allows a bee to codifies impossible solutions. Does not
    assures that a solution is found
    """

    def is_better(self, bee):
        """
        Selects the better solution following  Deb’s rules:
        • Any feasible solution (violationi ≤ 0) is preferred to any infeasible
        solution (violationj > 0) (solution i is dominant);
        • Among two feasible solutions(violationi ≤ 0, violationj ≤ 0), the
        one having better objective function value is preferred (fi < fj,
        solution i is dominant);
        • Among two infeasible solutions (violationi > 0, violationj >
        0), the one having smaller constraint violation is preferred
        (violationi < violationj , solution i is dominant).

        :param bee: other BeeContinuousConstrained bee
        :return: True if this bee is better than the given bee, False otherwise
        """
        better = False

        if self.violations == 0 and bee.violations == 0:
            better = self.fitness > bee.fitness
        elif self.violations == 0 and bee.violations > 0:
            better = True
        elif self.violations > 0 and bee.violations == 0:
            better = False
        elif self.violations > 0 and bee.violations > 0:
            better = self.violations < bee.violations

        return better

    def probability(self, hive_population):
        """
        Computes probabilities using the method described in the article:
        Karaboga, D., & Akay, B. (2011). A modified artificial bee colony
        (ABC) algorithm for constrained optimization problems. Applied soft computing, 11(3), 3021-3031

        infeasible solutions have probabilities in the range [0.0,0.5]. Feasible solutions have probabilities in the
        range [0.5,1.0]

        :param hive_population: list of BeeContinuousConstrained
        :return: real value between [0,1]
        """

        if self.violations == 0:
            hive_fitness = [bee.fitness for bee in hive_population]
            sum_fitness = sum(hive_fitness)
            probability = 0.5 + 0.5*(self.fitness/sum_fitness)
        else:
            hive_violations = [bee.violations for bee in hive_population]
            sum_violations = sum(hive_violations)
            probability = (1 - self.violations/sum_violations)*0.5

        return probability

    def randomize(self):
        """
        Transforms this bee's solution into a new random one, selecting a value between l
        ower and upper bound for each dimension
        :return: None
        """
        # creates a random solution vector
        self.vector = self._random()

        # check constraints
        self.violations = self.check_constraints(self.vector)

        # update fitness if feasible
        if self.violations == 0:
            # computes fitness of solution vector
            value = self.fitness_fun(self.vector)
            self.fitness = self._fitness(value)
            self.eval_value = value
        else:
            self.fitness = 0.0
            self.eval_value = float('nan')

        # initialises trial limit counter - i.e. abandonment counter
        self.counter = 0

    def crossover(self, bee, hive_state):
        """
        Generates a new Solution bee based on other bee. Uses the method described in the article:
        Karaboga, D., & Akay, B. (2011). A modified artificial bee colony
        (ABC) algorithm for constrained optimization problems. Applied soft computing, 11(3), 3021-3031

        :param bee: other bee, of the same class, to generate the solution with
        :return: a new bee that codifies the new solution generating from the input bees
        """
        zombee = self.clone()

        # select the dimensions to be mutated
        d = [0]*self.dim
        for i in range(self.dim):
            rj = random.uniform(0.0, 1.0)
            if rj < self.mr:
                d[i] = 1

        # produces a mutant based on current bee and bee's friend
        zombee._mutate(d, bee)

        # checks boundaries
        zombee._check()

        # check constraints
        zombee.violations = zombee.check_constraints(zombee.vector)

        # computes fitness if feasible
        if zombee.violations == 0:
            value = zombee.fitness_fun(zombee.vector)
            zombee.fitness = zombee._fitness(value)
            zombee.eval_value = value
        else:
            zombee.fitness = 0
            zombee.eval_value = float('nan')

        return zombee

    def __init__(self, lower, upper, fun, check_constraints, mr=0.5):
        """

        Instantiates a bee object randomly.

        Parameters:
        ----------
            :param list lower  : lower bound of solution vector
            :param list upper  : upper bound of solution vector
            :param def  fun    : evaluation function
            :param def check_constraints: return the number of constraints that a solution violates
            :param float mr: a real number between [0,1]. Controls the number of dimensions changed.
        """
        super().__init__(lower, upper, fun)

        self.check_constraints = check_constraints
        self.mr = mr
        self.violations = sys.maxsize

    def _mutate(self, dims, other_bee):
        """

        Mutates a given solution vector for continuous real-values.

        Parameters:
        ----------
            :param list dims                          : binary vector of dimensions to be mutated,
                                                        1 dimension mutates
                                                        0 otherwise
            :param BeeContinuousConstrained other_bee : other BeeContinuousConstrained object

        """
        for dim, flag in enumerate(dims):
            if flag is 1:
                phi = random.uniform(-1.0, 1.0)
                self.vector[dim] = self.vector[dim] + phi*(self.vector[dim] - other_bee.vector[dim])
