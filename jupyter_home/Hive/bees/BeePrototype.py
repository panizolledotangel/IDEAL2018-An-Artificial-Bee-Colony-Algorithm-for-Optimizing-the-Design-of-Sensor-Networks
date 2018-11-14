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
Interface of the Prototype object. All bees must implement this interface. This interface follows the Prototype 
design pattern (https://sourcemaking.com/design_patterns/prototype) and is used by the Hive to run the algorithm.
"""

# ---- Imports
import abc


# ---- BEE CLASS PROTOTYPE
class BeeProtype(object):
    """
    Interface of the Prototype object. All bees must implement this interface
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def clone(self):
        """
        Return a deep copy of this bee object
        :return: a new Bee object equals to this one
        """
        raise NotImplementedError("clone method needs to be implemented")

    @abc.abstractmethod
    def randomize(self):
        """
        Transform this bee to codify a new random solution. The fitness must be computed too
        :return: None
        """
        raise NotImplementedError("randomize method needs to be implemented")

    @abc.abstractmethod
    def crossover(self, bee, hive_state):
        """
        Generates a new solution using myself and an other bee of my type
        :param bee: the other bee to breed a new solution
        :param hive_state: actual state of the hive
        :return: a new Bee that codifies the new solution
        """
        raise NotImplementedError("crossover method needs to be implemented")

    @abc.abstractmethod
    def is_better(self, bee):
        """
        Checks if this bee is better that than the given bee
        :param bee: the other bee to compare with
        :return: True if this bee is better than the given bee, False otherwise
        """
        raise NotImplementedError("compare method needs to be implemented")

    @abc.abstractmethod
    def probability(self, hive_population):
        """
        Calculates the probability that this bee is selected
        :param hive_population: a population of bees of the same type as you
        :return: a real value between [0,1]
        """
        raise NotImplementedError("compare method needs to be implemented")

    @abc.abstractmethod
    def dimensions(self):
        """
        Returns the number of dimension problem that the bee codifies
        :return: the number of dimension of the problem as an integer
        """
        raise NotImplementedError("dimensions method needs to be implemented")
