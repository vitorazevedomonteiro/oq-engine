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

from openquake.hazardlib.imt import SA, PGA
from openquake.hazardlib.correlation.cross_spatial_correlation import (
    LothBaker2013CorrelationModel)
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
        corma = cormo.get_correlation_model(self.SITECOL, imts)
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
        cormo = LothBaker2013CorrelationModel(vs30_clustering=False)
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, PGA())
        aaae(lt, [[1.0,            0.0,            0.0],
                  [1.97514806e-02, 9.99804920e-01, 0.0],
                  [1.97514806e-02, 5.42206860e-20, 9.99804920e-01]])


class LothBaker2013ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = LothBaker2013CorrelationModel(vs30_clustering=False)
        intra_residuals_sampled = numpy.random.normal(size=(3, 100000))
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, PGA(), intra_residuals_sampled
        )
        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.002)

        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, PGA())
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol(self):
        filtered = self.SITECOL.filtered([0, 2])
        numpy.random.seed(13)
        cormo = LothBaker2013CorrelationModel(vs30_clustering=False)
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, PGA(), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
             [[-0.71239066, 0.75376638, -0.04450308, 0.45181234, 1.34510171],
              [0.51816327, 1.36481251, 0.86016437, 1.48732124, -1.01860545]],
             decimal=6)
