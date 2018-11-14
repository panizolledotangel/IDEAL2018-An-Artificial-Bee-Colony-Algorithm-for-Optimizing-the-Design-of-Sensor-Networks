#!/usr/bin/env python

# ---- MODULE DOCSTRING
import sys

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
artificial bee colony algorithm for binary optimization. Applied Soft Computing, 12(1), 342-352.' join with the one 
proposed in the article "Karaboga, D., & Akay, B. (2011). A modified artificial bee colony  (ABC) algorithm for 
constrained optimization problems. Applied soft computing, 11(3), 3021-3031". 

Its meant to be used in binary constrained environments
"""
import numpy as np

from Hive.bees.BeeBinary import BeeBinary


class BeeBinaryConstrained(BeeBinary):
    """ This bee works in binary constrained environments"""

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
            probability = 0.5 + 0.5 * (self.fitness / sum_fitness)
        else:
            hive_violations = [bee.violations for bee in hive_population]
            sum_violations = sum(hive_violations)
            probability = (1 - self.violations / sum_violations) * 0.5

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

        # check constraints
        zombee.violations = zombee.check_constraints(zombee.vector)

        # update fitness if feasible
        if zombee.violations == 0:
            # computes fitness of solution vector
            value = zombee.fitness_fun(zombee.vector)
            zombee.fitness = zombee._fitness(value)
            zombee.eval_value = value
        else:
            zombee.fitness = 0.0
            zombee.eval_value = float('nan')

        return zombee

    def __init__(self, dimensions, fun, check_constraints_fun, theta_min=0.5, theta_max=0.9):
        """
        Instantiates a bee object randomly.

        Parameters:
        ----------
            :param int dimensions  : number of dimensions the solution must have
            :param def  fun    : evaluation function
            :param def check_constraints_fun : function that return the number of constraints violated
            :param float theta_min : governs the minimum radius around a solutions being exploited
            :param float theta_max : governs the maximum radius around a solutions being exploited
        """
        # call super
        super().__init__(dimensions, fun, theta_min, theta_max)

        # checks input
        assert (check_constraints_fun is not None), "'check_constraints_fun' constraints function must not be None"

        self.check_constraints = check_constraints_fun
        self.violations = sys.maxsize




