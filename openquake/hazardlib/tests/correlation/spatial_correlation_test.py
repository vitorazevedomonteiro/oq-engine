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

from openquake.hazardlib.imt import SA, PGA, PGV
from openquake.hazardlib.correlation.spatial_correlation import (
    JB2009CorrelationModel,
    HM2018CorrelationModel,
    EI2012CorrelationModel,
    EI2011CorrelationModel,
    AHP2022CorrelationModel,
    S2022CorrelationModel,
    S2010CorrelationModel,
    SW2013CorrelationModel,
    DW2013CorrelationModel,
    
)
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Point

aaae = numpy.testing.assert_array_almost_equal


class JB2009CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_no_clustering(self):
        cormo = JB2009CorrelationModel(vs30_clustering=False)
        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,          0.03823366, 1,          0.03823366],
                     [0.03823366, 1,          0.03823366, 0.00146181],
                     [1,          0.03823366, 1,          0.03823366],
                     [0.03823366, 0.00146181, 0.03823366, 1]])

        imt = SA(period=0.95, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,          0.26107857, 1,          0.26107857],
                     [0.26107857, 1,          0.26107857, 0.06816202],
                     [1,          0.26107857, 1,          0.26107857],
                     [0.26107857, 0.06816202, 0.26107857, 1]])

    def test_clustered(self):
        cormo = JB2009CorrelationModel(vs30_clustering=True)
        imt = SA(period=0.001, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,          0.44046654, 1,          0.44046654],
                     [0.44046654, 1,          0.44046654, 0.19401077],
                     [1,          0.44046654, 1,          0.44046654],
                     [0.44046654, 0.19401077, 0.44046654, 1]])

        imt = SA(period=0.5, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,          0.36612758, 1,          0.36612758],
                     [0.36612758, 1,          0.36612758, 0.1340494],
                     [1,          0.36612758, 1,          0.36612758],
                     [0.36612758, 0.1340494, 0.36612758, 1]])

    def test_period_one_and_above(self):
        cormo = JB2009CorrelationModel(vs30_clustering=False)
        cormo2 = JB2009CorrelationModel(vs30_clustering=True)
        imt = SA(period=1.0, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,         0.2730787, 1,          0.2730787],
                     [0.2730787, 1,          0.2730787, 0.07457198],
                     [1,         0.2730787, 1,          0.2730787],
                     [0.2730787, 0.07457198, 0.2730787, 1]])
        corma2 = cormo2._get_correlation_matrix(self.SITECOL, imt)
        self.assertTrue((corma == corma2).all())

        imt = SA(period=10.0, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,          0.56813402, 1,          0.56813402],
                     [0.56813402, 1,          0.56813402, 0.32277627],
                     [1,          0.56813402, 1,          0.56813402],
                     [0.56813402, 0.32277627, 0.56813402, 1]])
        corma2 = cormo2._get_correlation_matrix(self.SITECOL, imt)
        self.assertTrue((corma == corma2).all())

    def test_pga(self):
        sa = SA(period=1e-50, damping=5)
        pga = PGA()

        cormo = JB2009CorrelationModel(vs30_clustering=False)
        corma = cormo._get_correlation_matrix(self.SITECOL, sa)
        corma2 = cormo._get_correlation_matrix(self.SITECOL, pga)
        self.assertTrue((corma == corma2).all())

        cormo = JB2009CorrelationModel(vs30_clustering=True)
        corma = cormo._get_correlation_matrix(self.SITECOL, sa)
        corma2 = cormo._get_correlation_matrix(self.SITECOL, pga)
        self.assertTrue((corma == corma2).all())


class JB2009LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = JB2009CorrelationModel(vs30_clustering=False)
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, PGA())
        aaae(lt, [[1.0,            0.0,            0.0],
                  [1.97514806e-02, 9.99804920e-01, 0.0],
                  [1.97514806e-02, 5.42206860e-20, 9.99804920e-01]])


class JB2009ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = JB2009CorrelationModel(vs30_clustering=False)
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
        cormo = JB2009CorrelationModel(vs30_clustering=False)
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, PGA(), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
             [[-0.71239066, 0.75376638, -0.04450308, 0.45181234, 1.34510171],
              [0.51816327, 1.36481251, 0.86016437, 1.48732124, -1.01860545]],
             decimal=6)


class HM2018CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_correlation_no_uncertainty(self):
        cormo = HM2018CorrelationModel(uncertainty_multiplier=0)

        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1.0000000,    0.3981537,    1.0000000,    0.3981537],
                     [0.3981537,    1.0000000,    0.3981537,    0.2596809],
                     [1.0000000,    0.3981537,    1.0000000,    0.3981537],
                     [0.3981537,    0.2596809,    0.3981537,    1.0000000]])

        imt = SA(period=0.5, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1.0000000,    0.3809173,    1.0000000,    0.3809173],
                     [0.3809173,    1.0000000,    0.3809173,    0.2433886],
                     [1.0000000,    0.3809173,    1.0000000,    0.3809173],
                     [0.3809173,    0.2433886,    0.3809173,    1.0000000]])

        imt = SA(period=1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1.0000000,    0.3906193,    1.0000000,    0.3906193],
                     [0.3906193,    1.0000000,    0.3906193,    0.2525181],
                     [1.0000000,    0.3906193,    1.0000000,    0.3906193],
                     [0.3906193,    0.2525181,    0.3906193,    1.0000000]])

        imt = SA(period=2, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1.0000000,    0.4011851,    1.0000000,    0.4011851],
                     [0.4011851,    1.0000000,    0.4011851,    0.2625807],
                     [1.0000000,    0.4011851,    1.0000000,    0.4011851],
                     [0.4011851,    0.2625807,    0.4011851,    1.0000000]])

        imt = SA(period=4, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1.0000000,    0.3522765,    1.0000000,    0.3522765],
                     [0.3522765,    1.0000000,    0.3522765,    0.2170695],
                     [1.0000000,    0.3522765,    1.0000000,    0.3522765],
                     [0.3522765,    0.2170695,    0.3522765,    1.0000000]])

        imt = SA(period=6, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1.0000000,    0.3159779,    1.0000000,    0.3159779],
                     [0.3159779,    1.0000000,    0.3159779,    0.1851206],
                     [1.0000000,    0.3159779,    1.0000000,    0.3159779],
                     [0.3159779,    0.1851206,    0.3159779,    1.0000000]])

    def test_correlation_small_uncertainty(self):
        imt = SA(period=1.5, damping=5)

        cormo = HM2018CorrelationModel(uncertainty_multiplier=0)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)

        cormo2 = HM2018CorrelationModel(uncertainty_multiplier=1E-30)
        corma2 = cormo2._get_correlation_matrix(self.SITECOL, imt)
        self.assertTrue((corma == corma2).all())

    def test_pga_no_uncertainty(self):
        sa = SA(period=1e-50, damping=5)
        pga = PGA()

        cormo = HM2018CorrelationModel(uncertainty_multiplier=0)
        corma = cormo._get_correlation_matrix(self.SITECOL, sa)
        corma2 = cormo._get_correlation_matrix(self.SITECOL, pga)
        self.assertTrue((corma == corma2).all())

    def test_correlation_with_uncertainty(self):
        Nsim = 100000
        cormo = HM2018CorrelationModel(uncertainty_multiplier=1)
        imt = SA(period=3, damping=5)

        corma_3d = numpy.zeros((len(self.SITECOL), len(self.SITECOL), Nsim))

        # For each simulation, construct a new correlation matrix
        for isim in range(0, Nsim):
            corma_3d[0:, 0:, isim] = \
                cormo._get_correlation_matrix(self.SITECOL, imt)

        # Mean and Coefficient of Variation (COV) of correlation matrix
        MEANcorMa = corma_3d.mean(2)
        COVcorma = numpy.divide(corma_3d.std(2), MEANcorMa)

        aaae(MEANcorMa, [[1.0000000, 0.3766436, 1.0000000, 0.3766436,],
                         [0.3766436, 1.0000000, 0.3766436, 0.2534904,],
                         [1.0000000, 0.3766436, 1.0000000, 0.3766436,],
                         [0.3766436, 0.2534904, 0.3766436, 1.00000,]], 2)

        aaae(COVcorma, [[0.0000000, 0.4102512, 0.0000000, 0.4102512,],
                        [0.4102512, 0.0000000, 0.4102512, 0.5636907,],
                        [0.0000000, 0.4102512, 0.0000000, 0.4102512,],
                        [0.4102512, 0.5636907, 0.4102512, 0.00000,]], 2)


class HM2018ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.95), 1, 1, 1)])

    def test_no_uncertainty(self):
        numpy.random.seed(1)
        Nsim = 100000
        imt = SA(period=2.0, damping=5)
        stddev_intra = numpy.array([0.5, 0.6, 0.7])
        cormo = HM2018CorrelationModel(uncertainty_multiplier=0)

        intra_residuals_sampled = numpy.random.multivariate_normal(
            numpy.zeros(3), numpy.diag(stddev_intra ** 2), Nsim).\
            transpose(1, 0)

        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imt, intra_residuals_sampled, stddev_intra)

        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean(1)
        std = intra_residuals_correlated.std(1)

        aaae(numpy.squeeze(numpy.asarray(mean)), numpy.zeros(3), 2)
        aaae(numpy.squeeze(numpy.asarray(std)), stddev_intra, 2)

        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(inferred_corrcoef, actual_corrcoef, 2)

    def test_with_uncertainty(self):
        numpy.random.seed(1)
        Nsim = 100000
        imt = SA(period=3.0, damping=5)
        stddev_intra = numpy.array([0.3, 0.6, 0.9])
        cormo = HM2018CorrelationModel(uncertainty_multiplier=1)

        intra_residuals_sampled = numpy.random.multivariate_normal(
            numpy.zeros(3), numpy.diag(stddev_intra ** 2), Nsim).\
            transpose(1, 0)

        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, imt, intra_residuals_sampled, stddev_intra)

        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean(1)
        std = intra_residuals_correlated.std(1)

        aaae(numpy.squeeze(numpy.asarray(mean)), numpy.zeros(3), 2)
        aaae(numpy.squeeze(numpy.asarray(std)), stddev_intra, 2)
        aaae(inferred_corrcoef,
             [[1., 0.3807, 0.5066],
              [0.3807, 1., 0.3075],
              [0.5066, 0.3075, 1.]], 2)


class EI2012CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_esm_database(self):
        cormo = EI2012CorrelationModel(database=1)
        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,         0.07638476, 1,         0.07638476],
                     [0.07638476, 1,         0.07638476, 0.00583463],
                     [1,         0.07638476, 1,         0.07638476],
                     [0.07638476, 0.00583463, 0.07638476, 1]])

        imt = SA(period=0.95, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,         0.24569092, 1,         0.24569092],
                     [0.24569092, 1,         0.24569092, 0.06036403],
                     [1,         0.24569092, 1,         0.24569092],
                     [0.24569092, 0.06036403, 0.24569092, 1]])

    def test_itaka_database(self):
        cormo = EI2012CorrelationModel(database=2)
        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,         0.03278196, 1,         0.03278196],
                     [0.03278196, 1,         0.03278196, 0.00107466],
                     [1,         0.03278196, 1,         0.03278196],
                     [0.03278196, 0.00107466, 0.03278196, 1]])

        imt = SA(period=0.5, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,         0.09861213, 1,         0.09861213],
                     [0.09861213, 1,         0.09861213, 0.00972435],
                     [1,         0.09861213, 1,         0.09861213],
                     [0.09861213, 0.00972435, 0.09861213, 1]])

    def test_period_one_and_above(self):
        cormo = EI2012CorrelationModel(database=1)
        cormo2 = EI2012CorrelationModel(database=2)
        imt = SA(period=1.0, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,         0.25483201, 1,         0.25483201],
                     [0.25483201, 1,         0.25483201, 0.06493935],
                     [1,         0.25483201, 1,         0.25483201],
                     [0.25483201, 0.06493935, 0.25483201, 1]])
        cormo2._get_correlation_matrix(self.SITECOL, imt)

        imt = SA(period=2.0, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1,         0.40691525, 1,         0.40691525],
                     [0.40691525, 1,         0.40691525, 0.16558002],
                     [1,         0.40691525, 1,         0.40691525],
                     [0.40691525, 0.16558002, 0.40691525, 1]])
        cormo2._get_correlation_matrix(self.SITECOL, imt)


class EI2012LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = EI2012CorrelationModel(database=1)
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, SA(0.5))
        aaae(lt, [[1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                  [1.57533818e-01,  9.87513593e-01,  0.00000000e+00],
                  [1.57533818e-01, -3.91284202e-18,  9.87513593e-01]])


class EI2012ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = EI2012CorrelationModel(database=1)
        intra_residuals_sampled = numpy.random.normal(size=(3, 100000))
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, SA(0.5), intra_residuals_sampled
        )
        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.002)

        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, SA(0.5))
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol(self):
        filtered = self.SITECOL.filtered([0, 2])
        numpy.random.seed(13)
        cormo = EI2012CorrelationModel(database=1)
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, SA(0.5), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
             [[-0.71239066, 0.75376638, -0.04450308, 0.45181234, 1.34510171],
              [0.41346528, 1.4520726, 0.8434472, 1.53139799, -0.82042512]],
             decimal=6)




class EI2011CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_esm_database(self):
        cormo = EI2011CorrelationModel(database=1)
        imt = PGA()
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.08450045, 1, 0.08450045],
                     [0.08450045, 1, 0.08450045, 0.00714033],
                     [1, 0.08450045, 1, 0.08450045],
                     [0.08450045, 0.00714033, 0.08450045, 1]])

        imt = PGV()
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.21191774, 1, .21191774],
                     [0.21191774, 1, 0.21191774, 0.04490913],
                     [1, 0.21191774, 1, 0.21191774],
                     [0.21191774, 0.04490913, 0.21191774, 1]])

    def test_itaka_database(self):
        cormo = EI2011CorrelationModel(database=2)
        imt = PGA()
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.05498267, 1, 0.05498267],
                     [0.05498267, 1, 0.05498267, 0.00302309],
                     [1, 0.05498267, 1, 0.05498267],
                     [0.05498267, 0.00302309, 0.05498267, 1]])

        imt = PGV()
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.10020024, 1, 0.10020024],
                     [0.10020024, 1, 0.10020024, 0.01004009],
                     [1, 0.10020024, 1, 0.10020024],
                     [0.10020024, 0.01004009, 0.10020024, 1]])

class EI2011LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = EI2011CorrelationModel(database=1)
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, PGA())
        aaae(lt, [[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [8.45004542e-02, 9.96423441e-01, 0.00000000e+00],
                  [8.45004542e-02, 2.32060495e-17, 9.96423441e-01]])

        cormo2 = EI2011CorrelationModel(database=2)
        lt = cormo2.get_lower_triangle_correlation_matrix(self.SITECOL, PGA())
        aaae(lt, [[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [5.49826710e-02, 9.98487309e-01, 0.00000000e+00],
                  [5.49826710e-02, -2.85792901e-17, 9.98487309e-01]])


class EI2011ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = EI2011CorrelationModel(database=1)
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
        cormo = EI2011CorrelationModel(database=1)
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, PGA(), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
             [[-0.71239066, 0.75376638, -0.04450308, 0.45181234, 1.34510171],
              [0.47023662, 1.40905247, 0.85437067, 1.51157548, -0.92797657]],
             decimal=6)





class AHP2022CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = AHP2022CorrelationModel()
        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.39669678, 1, 0.39669678],
                     [0.39669678, 1, 0.39669678, 0.24864585],
                     [1, 0.39669678, 1, 0.39669678],
                     [0.39669678, 0.24864585, 0.39669678, 1]])

        imt = SA(period=0.95, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.39326062, 1, 0.39326062],
                     [0.39326062, 1, 0.39326062, 0.24541103],
                     [1, 0.39326062, 1, 0.39326062],
                     [0.39326062, 0.24541103, 0.39326062, 1]])

    def test_pga(self):
        sa = SA(period=1e-50, damping=5)
        pga = PGA()

        cormo = AHP2022CorrelationModel()
        corma = cormo._get_correlation_matrix(self.SITECOL, sa)
        corma2 = cormo._get_correlation_matrix(self.SITECOL, pga)
        self.assertTrue((corma == corma2).all())

        corma = cormo._get_correlation_matrix(self.SITECOL, sa)
        corma2 = cormo._get_correlation_matrix(self.SITECOL, pga)
        self.assertTrue((corma == corma2).all())


class AHP2022LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = AHP2022CorrelationModel()
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, SA(1.0))
        aaae(lt, [[1, 0, 0],
                  [0.39326062, 0.91942704, 0],
                  [0.39326062, 0.09871051, 0.91411286]])


class AHP2022ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = AHP2022CorrelationModel()
        intra_residuals_sampled = numpy.random.normal(size=(3, 100000))
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, SA(1.0), intra_residuals_sampled
        )
        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.002)

        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, SA(1.0))
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol(self):
        filtered = self.SITECOL.filtered([0, 2])
        numpy.random.seed(13)
        cormo = AHP2022CorrelationModel()
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, SA(1.0), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
                [[-0.71239066, 0.75376638, -0.04450308,  0.45181234, 1.34510171],
                 [ 0.20646171, 1.53065076, 0.76974308, 1.52936565, -0.42661714]],
             decimal=6)
        
        


class S2022CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_northItaly_database(self):
        cormo = S2022CorrelationModel(region=1)
        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.51971599, 1, 0.51971599],
                     [0.51971599, 1, 0.51971599, 0.27010471],
                     [1, 0.51971599, 1, 0.51971599],
                     [0.51971599, 0.27010471, 0.51971599, 1]])

        imt = SA(period=0.95, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.37276002, 1, 0.37276002],
                     [0.37276002, 1, 0.37276002, 0.13895004],
                     [1, 0.37276002, 1, 0.37276002],
                     [0.37276002, 0.13895004, 0.37276002, 1]])

    def test_centralItaly_database(self):
        cormo = S2022CorrelationModel(region=2)
        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.27083945, 1, 0.27083945],
                     [0.27083945, 1, 0.27083945, 0.07335401],
                     [1, 0.27083945, 1, 0.27083945],
                     [0.27083945, 0.07335401, 0.27083945, 1]])

        imt = SA(period=0.95, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.16149741, 1, 0.16149741],
                     [0.16149741, 1, 0.16149741, 0.02608141],
                     [1, 0.16149741, 1, 0.16149741],
                     [0.16149741, 0.02608141, 0.16149741, 1]])
        
    def test_southItaly_database(self):
        cormo = S2022CorrelationModel(region=3)
        imt = SA(period=0.1, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.23012143, 1, 0.23012143],
                     [0.23012143, 1, 0.23012143, 0.05295587],
                     [1, 0.23012143, 1, 0.23012143],
                     [0.23012143, 0.05295587, 0.23012143, 1]])

        imt = SA(period=0.95, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.1580499, 1, 0.1580499],
                     [0.1580499, 1, 0.1580499, 0.02497977],
                     [1, 0.1580499, 1, 0.1580499],
                     [0.1580499, 0.02497977, 0.1580499, 1]])

    def test_pga(self):
        sa = SA(period=1e-50, damping=5)
        pga = PGA()

        cormo = S2022CorrelationModel(region=1)
        corma = cormo._get_correlation_matrix(self.SITECOL, sa)
        corma2 = cormo._get_correlation_matrix(self.SITECOL, pga)
        self.assertTrue((corma == corma2).all())

        corma = cormo._get_correlation_matrix(self.SITECOL, sa)
        corma2 = cormo._get_correlation_matrix(self.SITECOL, pga)
        self.assertTrue((corma == corma2).all())

class S2022LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = S2022CorrelationModel(region=1)
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, SA(0.5))
        aaae(lt, [[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [3.30012946e-01, 9.43976406e-01, 0.00000000e+00],
                  [3.30012946e-01, 1.16027066e-17, 9.43976406e-01]])


class S2022ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = S2022CorrelationModel(region=1)
        intra_residuals_sampled = numpy.random.normal(size=(3, 100000))
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, SA(0.5), intra_residuals_sampled
        )
        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.002)
        self.assertAlmostEqual(std, 1, delta=0.002)

        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, SA(0.5))
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol(self):
        filtered = self.SITECOL.filtered([0, 2])
        numpy.random.seed(13)
        cormo = S2022CorrelationModel(region=1)
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, SA(0.5), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
             [[-0.71239066, 0.75376638, -0.04450308, 0.45181234, 1.34510171],
              [ 0.26741627, 1.52329818, 0.79827663, 1.54494837, -0.54291037]],
             decimal=6)




class S2010CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = S2010CorrelationModel()
        imt = PGA()
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.29387078, 1, 0.29387078],
                     [0.29387078, 1, 0.29387078, 0.2200366],
                     [1, 0.29387078, 1, 0.29387078],
                     [0.29387078, 0.2200366, 0.29387078, 1]])


class S2010LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = S2010CorrelationModel()
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, PGA())
        aaae(lt, [[1, 0, 0],
                  [0.29387078, 0.95584516, 0],
                  [0.29387078, 0.13985169, 0.94555881]])



class S2010ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = S2010CorrelationModel()
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
        cormo = S2010CorrelationModel()
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, PGA(), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
             [[-0.71239066, 0.75376638, -0.04450308, 0.45181234, 1.34510171],
              [0.29400598, 1.49819198, 0.80124785, 1.53095877, -0.59317946]],
             decimal=6)




class SW2013CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_pga(self):
        cormo = SW2013CorrelationModel()
        imt = PGA()
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.61891268, 1, 0.61891268],
                     [0.61891268, 1, 0.61891268, 0.45480041],
                     [1, 0.61891268, 1, 0.61891268],
                     [0.61891268, 0.45480041, 0.61891268, 1]])
        
    def test_pgv(self):
        cormo = SW2013CorrelationModel()
        imt = PGV()
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.57512564, 1, 0.57512564],
                     [0.57512564, 1, 0.57512564, 0.38577656],
                     [1, 0.57512564, 1, 0.57512564],
                     [0.57512564, 0.38577656, 0.57512564, 1]])


class SW2013LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_pga(self):
        cormo = SW2013CorrelationModel()
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, PGA())
        aaae(lt, [[1, 0, 0],
                  [0.61891268, 0.7854598, 0],
                  [0.61891268, 0.09134459, 0.78013028]])

    def test_pgv(self):
        cormo = SW2013CorrelationModel()
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, PGV())
        aaae(lt, [[1, 0, 0],
                  [0.57512564, 0.8180651, 0],
                  [0.57512564, 0.06724044, 0.81529702]])


class SW2013ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = SW2013CorrelationModel()
        intra_residuals_sampled = numpy.random.normal(size=(3, 100000))
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, PGA(), intra_residuals_sampled
        )
        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.003)

        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, PGA())
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol(self):
        filtered = self.SITECOL.filtered([0, 2])
        numpy.random.seed(13)
        cormo = SW2013CorrelationModel()
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, PGA(), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
             [[-0.71239066, 0.75376638, -0.04450308,  0.45181234, 1.34510171],
              [-0.02561471, 1.51983804, 0.64431355, 1.43319991, 0.01697015]],
             decimal=6)


class DW2013CorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test_sa(self):
        cormo = DW2013CorrelationModel(beta_vs30=10)
        imt = SA(period=0.2, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.11461977, 1, 0.11461977],
                     [0.11461977, 1, 0.11461977, 0.01313769],
                     [1, 0.11461977, 1, 0.11461977],
                     [0.11461977, 0.01313769, 0.11461977, 1]])

        imt = SA(period=5.0, damping=5)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.4796166, 1, 0.4796166],
                     [0.4796166, 1, 0.4796166, 0.23003209],
                     [1, 0.4796166, 1, 0.4796166],
                     [0.4796166, 0.23003209, 0.4796166, 1]])
    
    def test_pga(self):
        imt = PGA()
        cormo = DW2013CorrelationModel(beta_vs30=10)
        corma = cormo._get_correlation_matrix(self.SITECOL, imt)
        aaae(corma, [[1, 0.10822594, 1, 0.10822594],
                     [0.10822594, 1, 0.10822594, 0.01171285],
                     [1, 0.10822594, 1, 0.10822594],
                     [0.10822594, 0.01171285, 0.10822594, 1]])


class DW2013LowerTriangleCorrelationMatrixTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        cormo = DW2013CorrelationModel(beta_vs30=15)
        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, SA(1.0))
        aaae(lt, [[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [3.83438148e-01, 9.23566558e-01, 0.00000000e+00],
                  [3.83438148e-01, -2.91378869e-18, 9.23566558e-01]])

        lt = cormo.get_lower_triangle_correlation_matrix(self.SITECOL, PGA())
        aaae(lt, [[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [2.08691524e-01, 9.77981517e-01, 0.00000000e+00],
                  [2.08691524e-01, 1.41715839e-18, 9.77981517e-01]])


class DW2013ApplyCorrelationTestCase(unittest.TestCase):
    SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                              Site(Point(2, -40.1), 1, 1, 1),
                              Site(Point(2, -39.9), 1, 1, 1)])

    def test(self):
        numpy.random.seed(13)
        cormo = DW2013CorrelationModel(beta_vs30=15)
        intra_residuals_sampled = numpy.random.normal(size=(3, 100000))
        intra_residuals_correlated = cormo.apply_correlation(
            self.SITECOL, SA(1.0), intra_residuals_sampled
        )
        inferred_corrcoef = numpy.corrcoef(intra_residuals_correlated)
        mean = intra_residuals_correlated.mean()
        std = intra_residuals_correlated.std()
        self.assertAlmostEqual(mean, 0, delta=0.003)
        self.assertAlmostEqual(std, 1, delta=0.003)

        actual_corrcoef = cormo._get_correlation_matrix(self.SITECOL, SA(1.0))
        numpy.testing.assert_almost_equal(inferred_corrcoef, actual_corrcoef,
                                          decimal=2)

    def test_filtered_sitecol(self):
        filtered = self.SITECOL.filtered([0, 2])
        numpy.random.seed(13)
        cormo = DW2013CorrelationModel(beta_vs30=15)
        intra_residuals_sampled = numpy.random.normal(size=(2, 5))
        intra_residuals_correlated = cormo.apply_correlation(
            filtered, SA(1.0), intra_residuals_sampled)
        aaae(intra_residuals_correlated,
                [[-0.71239066, 0.75376638, -0.04450308, 0.45181234, 1.34510171],
                 [0.21849171, 1.53601118, 0.77832185, 1.53890678, -0.44971205]],
             decimal=6)
        