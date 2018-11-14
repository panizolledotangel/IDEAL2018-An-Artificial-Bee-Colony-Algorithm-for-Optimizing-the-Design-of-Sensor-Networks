import unittest

import numpy as np
import pandas

from sources.problem_formulation.sensor_network_design_examples import example_1, case_2, case_3


class TestExamples(unittest.TestCase):

    def test_case_1(self):
        # load ground truth
        costo_df = pandas.read_csv('resources/ejemplo_1_costo.csv', header=None)
        d_df = pandas.read_csv('resources/ejemplo_1_d.csv', header=None)
        mpa_df = pandas.read_csv('resources/ejemplo_1_mpa.csv', header=None)
        req_df = pandas.read_csv('resources/ejemplo_1_req.csv', header=None)
        cotaexact_df = pandas.read_csv('resources/ejemplo_1_cotaexact.csv', header=None)

        # compare with example
        sndp = example_1()

        assert np.allclose(sndp.costo, costo_df.values, rtol=1e-04, atol=1e-04), "costo is not equal to ground truth"
        assert np.allclose(sndp.incidence_matrix, d_df.values, rtol=1e-04, atol=1e-04), "d is not equal to ground truth"
        assert np.allclose(sndp.absolute_precision_matrix, mpa_df.values, rtol=1e-04,
                           atol=1e-04), "mpa is not equal to ground truth"
        assert np.allclose(sndp.required + 1, req_df.values, rtol=1e-04, atol=1e-04), "req is not equal to ground truth"
        assert np.allclose(sndp.bounds, cotaexact_df.values, rtol=1e-04,
                           atol=1e-04), "cotaexact is not equal to ground truth"

    def test_case_2(self):
        # load ground truth
        costo_df = pandas.read_csv('resources/ejemplo_2_costo.csv', header=None)
        d_df = pandas.read_csv('resources/ejemplo_2_d.csv', header=None)
        mpa_df = pandas.read_csv('resources/ejemplo_2_mpa.csv', header=None)
        req_df = pandas.read_csv('resources/ejemplo_2_req.csv', header=None)
        cotaexact_df = pandas.read_csv('resources/ejemplo_2_cotaexact.csv', header=None)

        # compare with example
        sndp = case_2()

        assert np.allclose(sndp.costo, costo_df.values, rtol=1e-04, atol=1e-04), "costo is not equal to ground truth"
        assert np.allclose(sndp.incidence_matrix, d_df.values, rtol=1e-04, atol=1e-04), "d is not equal to ground truth"
        assert np.allclose(sndp.absolute_precision_matrix, mpa_df.values, rtol=1e-04, atol=1e-04), "mpa is not equal to ground truth"
        assert np.allclose(sndp.required + 1, req_df.values, rtol=1e-04, atol=1e-04), "req is not equal to ground truth"
        assert np.allclose(sndp.bounds, cotaexact_df.values, rtol=1e-04, atol=1e-04), "cotaexact is not equal to ground truth"

    def test_case_3(self):
        # load ground truth
        costo_df = pandas.read_csv('resources/ejemplo_3_costo.csv', header=None)
        d_df = pandas.read_csv('resources/ejemplo_3_d.csv', header=None)
        mpa_df = pandas.read_csv('resources/ejemplo_3_mpa.csv', header=None)
        req_df = pandas.read_csv('resources/ejemplo_3_req.csv', header=None)
        cotaexact_df = pandas.read_csv('resources/ejemplo_3_cotaexact.csv', header=None)

        # compare with example
        sndp = case_3()

        assert np.allclose(sndp.costo, costo_df.values, rtol=1e-04, atol=1e-04), "costo is not equal to ground truth"
        assert np.allclose(sndp.incidence_matrix, d_df.values, rtol=1e-04, atol=1e-04), "d is not equal to ground truth"
        assert np.allclose(sndp.absolute_precision_matrix, mpa_df.values, rtol=1e-04, atol=1e-04), "mpa is not equal to ground truth"
        assert np.allclose(sndp.required + 1, req_df.values, rtol=1e-04, atol=1e-04), "req is not equal to ground truth"
        assert np.allclose(sndp.bounds, cotaexact_df.values, rtol=1e-04, atol=1e-04), "cotaexact is not equal to ground truth"


if __name__ == '__main__':
    unittest.main()
