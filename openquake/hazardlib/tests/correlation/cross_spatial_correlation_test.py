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

from openquake.hazardlib.imt import SA
from openquake.hazardlib.correlation.cross_spatial_correlation import (
    LothBaker2013CorrelationModel, MarkhvidaEtAl2018CorrelationModel)
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


class LothBaker2013LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = LothBaker2013CorrelationModel()
        imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)]
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, imts)
        aaae(lt, [
            [0.99498744, 0, 0, 0, 0, 0],
            [0.32087354, 0.9418281, 0, 0, 0, 0],
            [0.32087354, 0.07432345, 0.93889094, 0, 0, 0],
            [0.33166248, 0.03488297, 0.03223073, 0.94220187, 0, 0],
            [0.13997712, 0.30269329, 0.01335984, 0.34923708, 0.87557875, 0],
            [0.13997712, 0.03720492, 0.30069504, 0.34923708, 0.06047969,
             0.87348747]
        ])


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
