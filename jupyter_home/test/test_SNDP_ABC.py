# ---- MODULE DOCSTRING
import unittest

# from Hive import Utilities
from sources.SensorNetworkDesignABC import SensorNetworkDesignABC
from sources.problem_formulation.sensor_network_design_examples import example_1


class TestSNDP(unittest.TestCase):

    def test_example_1(self):
        hive = SensorNetworkDesignABC(example_1(), verbose=True)
        _, best, _ = hive.run()

        assert best is not None

        # prints out best solution
        print("Solution ABC: {0}".format(best.vector))
        print("Constraints Violated: {0}".format(best.violations))
        print("Cost: {0}".format(best.eval_value))


if __name__ == '__main__':
    unittest.main()
