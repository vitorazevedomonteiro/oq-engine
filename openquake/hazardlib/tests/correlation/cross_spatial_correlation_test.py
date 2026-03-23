# The Hazard Library
# Copyright (C) 2012-2026 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy

from openquake.hazardlib.imt import SA, Sa_avg2, Sa_avg3, PGA, FIV3, CAV, PGV
from openquake.hazardlib.correlation.cross_spatial_correlation import (
    LothBaker2013CorrelationModel,
    WangDu2013CorrelationModel,
    MarkhvidaEtAl2018CorrelationModel,
    MonteiroEtAlGlobal2026CorrelationModel,
    DuNing2021CorrelationModel,
    MonteiroEtAlPairWise2026CorrelationModel)
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Point

aaae = numpy.testing.assert_array_almost_equal


class LothBaker2013CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = LothBaker2013CorrelationModel()
        imts = [SA(period=0.1, damping=5), SA(period=1.0, damping=5)]
        corma = cormo._get_correlation_matrix(self.SITECOL, imts)
        aaae(corma, [[1,    0.31737876, 1,  0.31737876, 0.2,    0.11200214,
                      0.2,    0.11200214],
                     [0.31737876,   1,  0.31737876, 0.1726039,
                         0.11200214, 0.2, 0.11200214,    0.06139007],
                     [1,    0.31737876, 1,  0.31737876, 0.2,
                         0.11200214, 0.2,    0.11200214],
                     [0.31737876, 0.1726039,  0.31737876, 1,
                         0.11200214, 0.06139007, 0.11200214, 0.2],
                     [0.2,  0.11200214, 0.2,    0.11200214,
                         1.01,   0.36029323, 1.01,   0.36029323],
                     [0.11200214,   0.2,    0.11200214, 0.06139007,
                         0.36029323, 1.01,   0.36029323, 0.19680408],
                     [0.2,  0.11200214, 0.2,    0.11200214,
                         1.01,   0.36029323, 1.01,   0.36029323],
                     [0.11200214,   0.06139007, 0.11200214, 0.2,    0.36029323,
                      0.19680408, 0.36029323, 1.01]])

        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]
        corma = cormo._get_correlation_matrix(self.SITECOL, imts)
        aaae(corma, [[0.99,       0.31926514, 0.99,       0.31926514, 0.33,
                      0.13927548,    0.33,       0.13927548],
                     [0.31926514, 0.99,       0.31926514, 0.17295974,
                         0.13927548, 0.33,   0.13927548, 0.0799556],
                     [0.99,       0.31926514, 0.99,       0.31926514,
                         0.33,       0.13927548, 0.33,       0.13927548],
                     [0.31926514, 0.17295974, 0.31926514, 0.99,
                         0.13927548, 0.0799556,  0.13927548, 0.33],
                     [0.33,       0.13927548, 0.33,       0.13927548,
                         1,         0.38646643, 1,         0.38646643],
                     [0.13927548, 0.33,       0.13927548, 0.0799556,
                         0.38646643, 1, 0.38646643, 0.20979378],
                     [0.33,       0.13927548, 0.33,       0.13927548,
                         1,         0.38646643, 1,         0.38646643],
                     [0.13927548, 0.0799556,  0.13927548, 0.33,
                      0.38646643, 0.20979378, 0.38646643, 1]])

class LothBaker2013ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = LothBaker2013CorrelationModel()

        # Two IMTs --> cross-IMT correlation
        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]
        num_sites = len(self.SITECOL)
        num_imts = len(imts)
        num_realizations = 100000

        # Sample uncorrelated residuals
        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites * num_imts, num_realizations))

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Flatten IMT x site for correlation computation
        flattened = intra_residuals_correlated.reshape(
            num_sites * num_imts, num_realizations)
        inferred_corrcoef = numpy.corrcoef(flattened)

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.002)

        # Compare to the correlation matrix from the model
        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, imts)
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol(self):
        filtered = self.SITECOL.filtered([0, 2])
        filtered.indices = numpy.array([0, 2])

        numpy.random.seed(13)
        cormo = LothBaker2013CorrelationModel()
        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]

        num_imts = len(imts)
        num_sites_filtered = len(filtered)
        num_realizations = 5

        # Sample uncorrrelated residuals
        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites_filtered * num_imts, num_realizations))
        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, imts, intra_residuals_sampled)

        # Assert expected correlated residuals
        aaae(intra_residuals_correlated, [
            [[-0.70881976, 0.74998808, -0.04428, 0.4495476, 1.3383593],
             [0.27121991, 1.50954287, 0.7943037, 1.53329927, -0.54988757]],
            [[-0.96250255, -0.89517391, 0.54331276, -0.03175413, 1.2733547],
             [0.0620108, 0.1821031, 2.32763212, 0.95248385, 0.16967118]],
        ],
            decimal=6
        )


class WangDu2013LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_sa_sa(self):
        cormo = WangDu2013CorrelationModel(Rvs30=12.5)
        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, imts)
        aaae(lt, [
            [1.0, 0, 0, 0, 0, 0],
            [0.25654938, 0.96653113, 0, 0, 0, 0],
            [0.25654938, 0.08330161, 0.96293471, 0, 0, 0],
            [0.29,       0.02454221, 0.02251077, 0.95644704, 0, 0],
            [0.09812013, 0.27399774, 0.01039717, 0.40868327, 0.8649668, 0],
            [0.09812013, 0.0339733,  0.27208211, 0.40868327, 0.0797542,
             0.86128209]
        ])

    def test_pga_pgv(self):
        cormo = WangDu2013CorrelationModel(Rvs30=12.5)
        imts = [PGA(), PGV()]
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, imts)
        aaae(lt, [
            [1.0, 0, 0, 0, 0, 0],
            [0.22385942,  0.97462144, 0, 0, 0, 0],
            [0.22385942,  0.06754524,  0.97227805, 0, 0, 0],
            [0.91,       -0.01021082, -0.00952607, 0.41437301, 0, 0],
            [0.19376039,  0.8891913,  -0.00410187,  0.03917511,  0.41260662, 0],
            [0.19376039,  0.05753257,  0.88733759,  0.03917511,  0.01157218,
             0.41244431]
        ])

class WangDu2013ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_sa_sa(self):
        numpy.random.seed(13)
        cormo = WangDu2013CorrelationModel(Rvs30=12.5)

        # Two IMTs --> cross-IMT correlation
        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]
        num_sites = len(self.SITECOL)
        num_imts = len(imts)
        num_realizations = 100000

        # Sample uncorrelated residuals
        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites * num_imts, num_realizations))

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Flatten IMT x site for correlation computation
        flattened = intra_residuals_correlated.reshape(
            num_sites * num_imts, num_realizations)
        inferred_corrcoef = numpy.corrcoef(flattened)

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.002)

        # Compare to the correlation matrix from the model
        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, imts)
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_pga_pgv(self):
        numpy.random.seed(13)
        cormo = WangDu2013CorrelationModel(Rvs30=12.5)

        # Two IMTs --> cross-IMT correlation
        imts = [PGA(), PGV()]
        num_sites = len(self.SITECOL)
        num_imts = len(imts)
        num_realizations = 100000

        # Sample uncorrelated residuals
        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites * num_imts, num_realizations))

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Flatten IMT x site for correlation computation
        flattened = intra_residuals_correlated.reshape(
            num_sites * num_imts, num_realizations)
        inferred_corrcoef = numpy.corrcoef(flattened)

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.002)

        # Compare to the correlation matrix from the model
        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, imts)
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol_sa_sa(self):
        filtered = self.SITECOL.filtered([0, 2])
        filtered.indices = numpy.array([0, 2])

        numpy.random.seed(13)
        cormo = WangDu2013CorrelationModel(Rvs30=12.5)
        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]

        num_imts = len(imts)
        num_sites_filtered = len(filtered)
        num_realizations = 5

        # Sample uncorrrelated residuals
        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites_filtered * num_imts, num_realizations))
        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, imts, intra_residuals_sampled)

        # Assert expected correlated residuals
        aaae(intra_residuals_correlated, [
            [[-0.71239066,  0.75376638, -0.04450308,  0.45181234,  1.34510171],
             [ 0.32984325,  1.49352109,  0.81787309,  1.53979   , -0.66154492]],
            [[-0.94923617, -0.95767325,  0.54481378, -0.06841674,  1.24049184],
             [0.0258218 , 0.03536842, 2.31206596, 0.86939802, 0.19792573]],
        ],
            decimal=6
        )

    def test_filtered_sitecol_pga_pgv(self):
        filtered = self.SITECOL.filtered([0, 2])
        filtered.indices = numpy.array([0, 2])

        numpy.random.seed(13)
        cormo = WangDu2013CorrelationModel(Rvs30=12.5)
        imts = [PGA(), PGV()]

        num_imts = len(imts)
        num_sites_filtered = len(filtered)
        num_realizations = 5

        # Sample uncorrrelated residuals
        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites_filtered * num_imts, num_realizations))
        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, imts, intra_residuals_sampled)

        # Assert expected correlated residuals
        aaae(intra_residuals_correlated, [
            [[-0.71239066,  0.75376638, -0.04450308,  0.45181234,  1.34510171],
             [ 0.35810508,  1.48149576,  0.82737448,  1.53883613, -0.71528354]],
            [[-0.98028235,  0.15028997,  0.18452676,  0.29623533,  1.61263038],
            [ 0.43431117,  1.34720451,  1.6645251 ,  1.64016475, -0.64222091]],
        ],
            decimal=6
        )

class MarkhvidaEtAl2018LowerTriangleCorrelationMatrixTestCase(
        unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = MarkhvidaEtAl2018CorrelationModel(num_pcs=5)
        imts = [SA(period=0.5, damping=5), SA(period=3.0, damping=5)]
        cormo.get_lower_triangle_correlation_matrix(self.SITECOL, imts)
        lt = cormo.cache["corma"]
        aaae(lt, [
            [[3.81103035, 2.17625169, 1.12381154, 0.79465476, 0.57516197],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[1.7735324,  1.04485274, 0.52152147, 0.23627793, 0],
             [3.37320844, 1.90901916, 0.99547363, 0.75871532, 0.57516197],
             [0, 0, 0, 0, 0]],
            [[1.7735324, 1.04485274, 0.52152147, 0.23627793, 0],
             [0.70395373, 0.37376553, 0.17093491, 0.10581377, 0],
             [3.29893685, 1.87207198, 0.98068802, 0.75130046, 0.57516197]]
        ]
        )


class MarkhvidaEtAl2018ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = MarkhvidaEtAl2018CorrelationModel(num_pcs=5)

        # Two IMTs --> cross-IMT correlation
        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.06)


class MonteiroEtAlGlobal2026LowerTriangleCorrelationMatrixTestCase(
        unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_pca_lower_triangle_corr_matrix(self):
        cormo = MonteiroEtAlGlobal2026CorrelationModel(num_pcs=3)
        imts = [SA(period=1.5, damping=5), SA(
            period=3.0, damping=5)]  # does not depend on IMs
        cormo.get_lower_triangle_correlation_matrix(self.SITECOL, imts)
        lt = cormo.cache["corma"]
        aaae(lt, [[[4.71568827, 2.19882828, 1.18072932],
                   [0, 0, 0],
                   [0, 0, 0]],
                  [[2.04622712, 1.20726616, 0.66510701],
                   [4.24860806, 1.83775793, 0.97557901],
                   [0, 0, 0]],
                  [[2.04622712, 1.20726616, 0.66510701],
                   [0.60807409, 0.24427623, 0.16102439],
                   [4.20486817, 1.82145089, 0.96219829]]]
             )


class MonteiroEtAlGlobal2026ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_sa_fiv3(self):
        numpy.random.seed(13)
        cormo = MonteiroEtAlGlobal2026CorrelationModel(num_pcs=3)

        # Two IMTs --> cross-IMT correlation
        imts = [SA(period=0.5, damping=5), FIV3(period=3.0, damping=5)]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.07)

    def test_saavg2_pga(self):
        numpy.random.seed(13)
        cormo = MonteiroEtAlGlobal2026CorrelationModel(num_pcs=3)

        # Two IMTs --> cross-IMT correlation
        imts = [Sa_avg2(period=0.5, damping=5), PGA()]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.12)


class DuNing2021LowerTriangleCorrelationMatrixTestCase(
        unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_pca_lower_triangle_corr_matrix(self):
        cormo = DuNing2021CorrelationModel(num_pcs=7)
        imts = [SA(period=1.5, damping=5), SA(
            period=3.0, damping=5)]  # does not depend on IMs
        cormo.get_lower_triangle_correlation_matrix(self.SITECOL, imts)
        lt = cormo.cache["corma"]
        aaae(lt, [
            [[3.65452535, 2.29249985, 1.58113883, 1.27366488, 0.98319208,
              0.80966385, 0.72264945],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]],
            [[2.63054635, 1.23738716, 1.10931976, 0.81803787, 0.44626876,
              0.46728999, 0.40115419],
             [2.5368842,  1.92987786, 1.12668082, 0.97623576,
              0.87607697, 0.6612077, 0.60108031],
             [0, 0, 0, 0, 0, 0, 0]],
            [[2.63054635, 1.23738716, 1.10931976, 0.81803787, 0.44626876,
              0.46728999, 0.40115419],
             [0.44885041, 0.24005588, 0.16936716, 0.134822,
              0.14323908, 0.1352941, 0.10667158],
             [2.49686098, 1.91488948, 1.11387811, 0.96688122, 0.86428781,
              0.64721799, 0.59153927]]]
             )


class DuNing2021CorrelationModelApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_sa_sa(self):
        numpy.random.seed(13)
        cormo = DuNing2021CorrelationModel(num_pcs=7)

        # Two IMTs --> cross-IMT correlation
        imts = [SA(period=0.5, damping=5), SA(period=3.0, damping=5)]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.005)
        self.assertAlmostEqual(std, 1, delta=0.07)

    def test_sa_pga(self):
        numpy.random.seed(13)
        cormo = DuNing2021CorrelationModel(num_pcs=7)

        # Two IMTs --> cross-IMT correlation
        imts = [SA(period=0.5, damping=5), PGA()]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.07)

    def test_sa_cav(self):
        numpy.random.seed(13)
        cormo = DuNing2021CorrelationModel(num_pcs=7)

        # Two IMTs --> cross-IMT correlation
        imts = [SA(period=0.5, damping=5), CAV()]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.07)


class MonteiroEtAlPairWise2026CorrelationModelApplyCorrelationTestCase(
        unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_lower_triangle_corr_matrix(self):
        cormo = MonteiroEtAlPairWise2026CorrelationModel()
        imts = [SA(1.5), SA(3.0)]  # does not depend on IMs
        cormo.get_lower_triangle_correlation_matrix(self.SITECOL, imts)
        lt = cormo.cache["corma"]
        aaae(lt, [[[1.48350454, 0.7475819],
                   [0, 0],
                   [0, 0]],
                  [[0.65657777, 0.42579227],
                   [1.33029746, 0.6144751],
                   [0, 0]],
                  [[0.65657777, 0.42579227],
                   [0.16754674, 0.07409421],
                   [1.3197043, 0.60999155]]]
             )


class MonteiroEtAlPairWise2026LowerTriangleCorrelationMatrixTestCase(
        unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_sa_sa(self):
        numpy.random.seed(13)
        cormo = MonteiroEtAlPairWise2026CorrelationModel()

        # Two IMTs --> cross-IMT correlation
        imts = [SA(0.5), SA(3.0)]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.005)
        self.assertAlmostEqual(std, 1, delta=0.12)

    def test_sa_pga(self):
        numpy.random.seed(13)
        cormo = MonteiroEtAlPairWise2026CorrelationModel()

        # Two IMTs --> cross-IMT correlation
        imts = [SA(0.5), PGA()]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.1)

    def test_sa_fiv3(self):
        numpy.random.seed(13)
        cormo = MonteiroEtAlPairWise2026CorrelationModel()

        # Two IMTs --> cross-IMT correlation
        imts = [SA(0.5), FIV3(1.0)]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.1)

    def test_saavg2_saavg3(self):
        numpy.random.seed(13)
        cormo = MonteiroEtAlPairWise2026CorrelationModel()

        # Two IMTs --> cross-IMT correlation
        imts = [Sa_avg2(0.5), Sa_avg3(0.9)]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.14)

    def test_fiv3_saavg2(self):
        numpy.random.seed(13)
        cormo = MonteiroEtAlPairWise2026CorrelationModel()

        # Two IMTs --> cross-IMT correlation
        imts = [FIV3(0.5), Sa_avg2(1.0)]
        num_sites = len(self.SITECOL)
        num_realizations = 100000

        # Sample uncorrelated residuals
        npcs = cormo.npcs

        intra_residuals_sampled = numpy.random.normal(
            size=(num_sites, num_realizations, npcs)
        )

        # Apply correlation
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imts, intra_residuals_sampled
        )

        # Check mean and std of correlated residuals
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.14)
