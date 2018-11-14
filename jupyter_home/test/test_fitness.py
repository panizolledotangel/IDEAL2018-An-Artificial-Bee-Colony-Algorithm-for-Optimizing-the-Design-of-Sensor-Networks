import unittest

import numpy as np
import pandas

from sources.problem_formulation.sensor_network_design_examples import example_1


class TestFitness(unittest.TestCase):

    def test_example_1(self):
        # load ground truth
        gt_df = pandas.read_csv('resources/ejemplo_1_results.csv', header=None)

        # creates model
        problem = example_1()
        sol_size = problem.incidence_matrix.shape[1]
        ndp = problem.dependence_matrix.shape[1]

        # start test
        gt = gt_df.values
        n_tests = gt.shape[0]

        for k in range(n_tests):

            # extract solution from ground truth vector
            combination = gt[k, 0:ndp]
            gt_flows = gt[k, ndp:2*ndp]
            gt_sal = gt[k, 2*ndp]
            gt_salob = gt[k, 2*ndp + 1:3*ndp + 1]
            gt_contrainsts_met = gt[k, 3*ndp + 1:4*ndp + 1]

            individuo = np.ones(sol_size)
            individuo[problem.dependence_matrix[0, :].astype(int)] = combination

            flows, sal, salob, constraints_met = problem.analyze_solution(individuo)

            assert np.allclose(np.abs(gt_flows - flows), np.zeros(ndp), rtol=1e-04, atol=1e-04), "fail in flows case {0}".format(k)
            assert int(gt_sal) == sal, "fail sal in case {0}".format(k)
            assert np.array_equal(gt_salob, salob), "fail in salob case {0}".format(k)
            assert np.array_equal(gt_contrainsts_met, constraints_met), "fail in constraints case {0}".format(k)


if __name__ == '__main__':
    unittest.main()
