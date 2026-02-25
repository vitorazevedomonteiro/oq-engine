# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2021, GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import logging
from copy import deepcopy
from scipy.interpolate import RectBivariateSpline
from abc import abstractmethod
from typing import Dict, List, Optional
from pathlib import Path
import h5py

from openquake.hazardlib.site import SiteCollection
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib.correlation.spatial_correlation import \
    BaseCorrelationModel


class BaseSpatialCrossCorrelationModel(BaseCorrelationModel):
    """
    Base class for cross-IM spatial correlation models for
    spatially-distributed ground-shaking intensities.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.cache = {"corma": None}

    def _get_correlation_matrix(self, sites: SiteCollection, imts: List):
        """
        Setup the correlation matrix given the sites and IMTs
        """
        distances = sites.mesh.get_distance_matrix()
        return self.get_correlation_model(distances, imts)

    @abstractmethod
    def get_correlation_model(self, distances: np.ndarray, imts: List):
        """
        Builds the correlation model - specific to the actual model itself
        """

    def get_lower_triangle_correlation_matrix(
            self, sites: SiteCollection, imts: List):
        # Apply Cholesky factorisation and retreive the
        # lower triangular correlation matrix
        self.cache["corma"] = np.linalg.cholesky(
            self._get_correlation_matrix(sites, imts))
        return self.cache["corma"]

    def apply_correlation(
            self, sites: SiteCollection, imts: List, residuals: np.ndarray):
        """
        Applies the correlation model to the sites
        """
        if not self.cache["corma"]:
            # For first implementation then get the lower
            # matrix
            logging.info("--- Building lower triangle correlation matrix")
            self.get_lower_triangle_correlation_matrix(sites.complete, imts)
            logging.info("--- --- done!")
        nsites = len(sites)
        logging.info("--- Generating spatially cross-correlated residuals")
        if len(sites.complete) == nsites:
            # No filtering of sites
            corr_residuals = np.matmul(self.cache["corma"], residuals)
        else:
            # Need to locate indices of correlation matrix corresponding
            # to the selected sites
            idx = np.tile(sites.indices, len(imts))
            lb = 0
            ub = len(sites)
            norig_sites = len(sites.complete)
            for i in range(len(imts)):
                idx[lb:ub] += i * norig_sites
                lb += nsites
                ub += nsites
            corr_residuals = np.matmul(
                self.cache["corma"][np.ix_(idx, idx)], residuals)

        # corr_residuals is 2-D array [Nimts * Nsites, Nevents]
        # need to re-arrange back to 3-D shape
        residuals = np.empty([len(imts), nsites, corr_residuals.shape[1]])
        idx = np.arange(nsites)
        for i in range(len(imts)):
            residuals[i] = corr_residuals[idx, :]
            idx += nsites
        logging.info("--- --- done!")
        return residuals

    def sample(self,
               num_realizations: int,
               sites: SiteCollection,
               imts: List,
               rng: Optional[np.random.Generator] = None,
               ) -> np.ndarray:
        """Generate random fields from the distribution

        :param num_realizations:
            Number of ground motion fields to generate
        :param sites:
            Site model as instance of
            `class`::shakyground2.site_model.SiteModel
        :param imts:
            List of intensity measure types (as string)
        :param rng:
            Seeded numpy random number generator (or None - uses random seed)

        :Returns: Sample fields of dimension
            [No. Intensiy Measures, No. Sites, No. Realizations]
        """
        if not rng:
            rng = np.random.default_rng()
        uncorrelated_residuals = rng.normal(
            0.0, 1.0, [len(sites) * len(imts), num_realizations]
        )

        return self.apply_correlation(sites, imts, uncorrelated_residuals)


class LothBaker2013CorrelationModel(BaseSpatialCrossCorrelationModel):
    """Implements the spatial cross-correlation model of Loth & Baker (2013)

    Loth, C., and Baker, J. W. (2013) A spatial cross-correlation model of
    spectral accelerations at multiple periods. Earthquake Engineering &
    Structural Dynamics, 42: 397 - 417

    Valid for periods 0.01 to 10.0 s
    """

    T = np.array([0.01, 0.1, 0.2, 0.5, 1.00, 2.00, 5.00, 7.5, 10.00])

    # Table II. Short range coregionalization matrix, B1
    B1 = np.array(
        [
            [0.29, 0.25, 0.23, 0.23, 0.18, 0.10, 0.06, 0.06, 0.06],
            [0.25, 0.30, 0.20, 0.16, 0.10, 0.04, 0.03, 0.04, 0.05],
            [0.23, 0.20, 0.27, 0.18, 0.10, 0.03, 0.00, 0.01, 0.02],
            [0.23, 0.16, 0.18, 0.31, 0.22, 0.14, 0.08, 0.07, 0.07],
            [0.18, 0.10, 0.10, 0.22, 0.33, 0.24, 0.16, 0.13, 0.12],
            [0.10, 0.04, 0.03, 0.14, 0.24, 0.33, 0.26, 0.21, 0.19],
            [0.06, 0.03, 0.00, 0.08, 0.16, 0.26, 0.37, 0.30, 0.26],
            [0.06, 0.04, 0.01, 0.07, 0.13, 0.21, 0.30, 0.28, 0.24],
            [0.06, 0.05, 0.02, 0.07, 0.12, 0.19, 0.26, 0.24, 0.23]
        ]
    )

    # Table III. Long range coregionalization matrix, B2
    B2 = np.array(
        [
            [0.47, 0.40, 0.43, 0.35, 0.27, 0.15, 0.13, 0.09, 0.12],
            [0.40, 0.42, 0.37, 0.25, 0.15, 0.03, 0.04, 0.00, 0.03],
            [0.43, 0.37, 0.45, 0.36, 0.26, 0.15, 0.09, 0.05, 0.08],
            [0.35, 0.25, 0.36, 0.42, 0.37, 0.29, 0.20, 0.16, 0.16],
            [0.27, 0.15, 0.26, 0.37, 0.48, 0.41, 0.26, 0.21, 0.21],
            [0.15, 0.03, 0.15, 0.29, 0.41, 0.55, 0.37, 0.33, 0.32],
            [0.13, 0.04, 0.09, 0.20, 0.26, 0.37, 0.51, 0.49, 0.49],
            [0.09, 0.00, 0.05, 0.16, 0.21, 0.33, 0.49, 0.62, 0.60],
            [0.12, 0.03, 0.08, 0.16, 0.21, 0.32, 0.49, 0.60, 0.68]
        ]
    )

    # Table IV. Nugget effect coregionalization matrix, B3
    B3 = np.array(
        [
            [0.24, 0.22, 0.21, 0.09, -0.02, 0.01, 0.03, 0.02, 0.01],
            [0.22, 0.28, 0.20, 0.04, -0.05, 0.00, 0.01, 0.01, -0.01],
            [0.21, 0.20, 0.28, 0.05, -0.06, 0.00, 0.04, 0.03, 0.01],
            [0.09, 0.04, 0.05, 0.26, 0.14, 0.05, 0.05, 0.05, 0.04],
            [-0.02, -0.05, -0.06, 0.14, 0.20, 0.07, 0.05, 0.05, 0.05],
            [0.01, 0.00, 0.00, 0.05, 0.07, 0.12, 0.08, 0.07, 0.06],
            [0.03, 0.01, 0.04, 0.05, 0.05, 0.08, 0.12, 0.10, 0.08],
            [0.02, 0.01, 0.03, 0.05, 0.05, 0.07, 0.10, 0.10, 0.09],
            [0.01, -0.01, 0.01, 0.04, 0.05, 0.06, 0.08, 0.09, 0.09]
        ]
    )

    def __repr__(self):
        return "LothBaker2013"

    def get_correlation_model(self, distances: np.ndarray, imts: List):
        """
        Build the correlation model for the particular
        configuration
        """
        periods = []
        for imt in imts:
            if str(imt) == "PGA":
                periods.append(0.01)
            elif "SA" in str(imt):
                if (imt.period > 10.0) or (imt.period < 0.01):
                    raise ValueError(
                        "Period %s out of range for Loth & Baker LMCR" % str(
                            imt.period)
                    )
                periods.append(imt.period)
            else:
                raise ValueError(
                    "Loth & Baker LMCR not supported for %s" % str(imt))

        periods = np.array(periods)
        b1mat = self.interp_matrix(periods, self.T, self.B1)
        b2mat = self.interp_matrix(periods, self.T, self.B2)
        b3mat = self.interp_matrix(periods, self.T, self.B3)
        dh = np.array(-3.0 * distances)
        if isinstance(dh, float):
            rho = b1mat * np.exp(dh / 20.0) + b2mat * np.exp(dh / 70.0)
            if dh < 1.0e-7:
                rho += b3mat
            return rho
        else:
            # dh is a matrix
            x, y = distances.shape
            mask = np.array(distances < 1.0e-7)
            nimts = len(periods)
            rho = np.zeros([nimts * x, nimts * y])
            idx_x = np.arange(0, x)
            for i in range(nimts):
                idx_y = np.arange(0, y)
                for j in range(nimts):
                    dummy = b1mat[i, j] * \
                        np.exp(dh / 20.0) + b2mat[i, j] * np.exp(dh / 70.0)
                    dummy[mask] += b3mat[i, j]
                    rho[np.ix_(idx_x, idx_y)] += dummy
                    idx_y += y
                idx_x += x
        return rho

    @staticmethod
    def interp_matrix(targets, periods, matrix):
        """Apply 2D interpolation to retrieve the values of the coefficient
        matrices at the required periods
        """
        f = RectBivariateSpline(periods, periods, matrix.T, kx=1, ky=1)
        return f(targets, targets).T


def get_isotropic_nested_cov(var_model: Dict, dist: np.ndarray) -> np.ndarray:
    """Returns the covariance matrix for the isotropic, nested, exponential
    semivariogram

    Args:
        var_model: Coefficients of the semivariogram model
        dist: distance matrix

    Returns:
        cov: Covariance matrix
    """
    var = var_model["Cn"] + var_model["C1"] + var_model["C2"]
    cov = var - (
        var_model["Cn"]
        + var_model["C1"] * (1.0 - np.exp(-3.0 * dist / var_model["A1"]))
        + var_model["C2"] * (1.0 - np.exp(-3.0 * dist / var_model["A2"]))
    )
    cov[dist == 0.0] = var
    return cov


def get_nugget_cov(var_model: Dict, dist: np.ndarray) -> np.ndarray:
    """Returns the covarance matrix for the nugget semivariogram

    Args:
        var_model: Coefficients of the semivariogram model
        dist: distance matrix

    Returns:
        cov: Covariance matrix
    """
    return np.eye(len(dist)) * var_model["Cn"]


class MarkhvidaEtAl2018CorrelationModel(BaseSpatialCrossCorrelationModel):
    """
    Implements the spatial cross-correlation model of Markhvida et al. (2018)
    based on principal component analysis and geostatistics.

    Markhvida, M., Ceferino, L., & Baker, J. W. (2018).
    Modeling spatially correlated spectral accelerations at multiple periods
    using principal component analysis and geostatistics. Earthquake
    Engineering & Structural Dynamics, 47(5), 1107-1123.
    https://doi.org/10.1002/eqe.3007
    """

    T = np.array(
        [
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.75,
            1.00,
            1.50,
            2.00,
            3.00,
            4.00,
            5.00,
        ]
    )

    PCA_COEFFS = CoeffsTable(
        sa_damping=5,
        table="""\
        imt                    1               2               3               4               5               6               7               8               9              10              11              12              13              14              15              16              17              18              19
        pga       2.70963956e-01 -1.39418157e-01  6.90420061e-02 -1.06094866e-01 -9.22880748e-02 -1.13489976e-01 -1.88935371e-01  1.53956802e-01 -1.60082932e-01 -4.85878662e-02  1.06169114e-01  5.45367125e-02 -8.42347289e-02  2.06507178e-03  2.33666516e-01 -4.44106081e-02 -2.98766213e-01 -5.27588860e-01 -5.80349073e-01
        0.010     2.70963956e-01 -1.39418157e-01  6.90420061e-02 -1.06094866e-01 -9.22880748e-02 -1.13489976e-01 -1.88935371e-01  1.53956802e-01 -1.60082932e-01 -4.85878662e-02  1.06169114e-01  5.45367125e-02 -8.42347289e-02  2.06507178e-03  2.33666516e-01 -4.44106081e-02 -2.98766213e-01 -5.27588860e-01 -5.80349073e-01
        0.020     2.70185457e-01 -1.41734439e-01  7.70156687e-02 -1.16393534e-01 -1.03464378e-01 -1.24082463e-01 -1.99840301e-01  1.55452551e-01 -1.57024101e-01 -5.11781532e-02  1.02685985e-01  5.34091782e-02 -7.85807733e-02  5.38047461e-03  2.20317829e-01 -3.94525931e-02 -2.57172669e-01 -1.50994889e-01  7.81868928e-01
        0.030     2.66716484e-01 -1.50918021e-01  1.01241750e-01 -1.44620230e-01 -1.28327845e-01 -1.50413273e-01 -2.17519115e-01  1.54533128e-01 -1.44555133e-01 -4.93784913e-02  8.65380809e-02  3.69034473e-02 -5.51975904e-02  7.87249482e-03  1.49651510e-01 -2.32087970e-02 -2.84597880e-02  8.08901724e-01 -2.26437335e-01
        0.050     2.51688240e-01 -1.84642999e-01  1.78879968e-01 -2.21328311e-01 -1.75557526e-01 -1.76668887e-01 -1.88651351e-01  4.24749059e-02 -4.55090102e-02 -2.91885727e-02 -3.15764521e-02 -6.05411455e-02  9.35508236e-02  2.24423933e-02 -2.99181352e-01  5.99168859e-02  7.54350403e-01 -2.06472973e-01  2.31093445e-02
        0.075     2.36434541e-01 -2.18922079e-01  2.37254184e-01 -2.34559034e-01 -1.33267088e-01 -4.31828094e-02  1.19447151e-01 -2.72310555e-01  2.38192698e-01  1.00676333e-01 -2.63315034e-01 -1.21207177e-01  2.02769694e-01  6.61936094e-03 -4.93306767e-01  1.16246173e-01 -4.75923823e-01  3.67733765e-02 -6.43955732e-03
        0.100     2.32994643e-01 -2.28087987e-01  2.30554573e-01 -1.60443024e-01  4.00564219e-02  1.81657485e-01  4.27112684e-01 -3.24579280e-01  2.63780433e-01  1.42634796e-01 -8.13780348e-02  4.65305509e-02 -1.51801547e-01 -8.33183140e-02  5.34198178e-01 -1.84596750e-01  2.10357917e-01 -2.84225564e-03  3.31108667e-03
        0.150     2.38919759e-01 -2.11905954e-01  1.32646222e-01  8.20453503e-02  3.27946973e-01  3.93273105e-01  3.25836316e-01  1.62029625e-01 -1.82164846e-01 -1.38319895e-01  4.70111475e-01  1.77876088e-01 -1.11256500e-01  8.83177907e-02 -2.91143253e-01  2.62494245e-01 -1.52291509e-03  1.54770016e-02  1.50080927e-03
        0.200     2.47247201e-01 -1.74053610e-01 -8.25743328e-03  2.77382297e-01  4.03271334e-01  2.20437620e-01 -8.37312940e-02  2.24796020e-01 -1.71941473e-01 -2.92747130e-02 -3.81524121e-01 -2.37244496e-01  3.56271009e-01 -8.50494997e-02 -1.25434811e-02 -4.42559355e-01  1.51125188e-02  1.10964548e-02  1.94187520e-03
        0.250     2.53677097e-01 -1.22375885e-01 -1.48595586e-01  3.65271223e-01  2.53186444e-01 -6.12442511e-02 -2.83389735e-01 -8.11437932e-02  2.12210668e-01  1.43362427e-01 -2.75727619e-01 -4.11307165e-02 -2.02014525e-01  2.28459039e-02  1.55257214e-01  6.32145332e-01  4.55130188e-02  6.90596762e-04  4.69634929e-04
        0.300     2.54921192e-01 -7.13194464e-02 -2.37030888e-01  3.59073100e-01  4.01080107e-02 -2.48766602e-01 -1.41859038e-01 -2.86692239e-01  3.00971238e-01  5.79993716e-02  3.28411992e-01  2.08361703e-01 -1.94768508e-01  3.24295009e-02 -2.58822161e-01 -4.77244327e-01  1.91094744e-03  6.62185259e-03  1.88266913e-04
        0.400     2.52458254e-01  1.25091294e-02 -3.27121081e-01  2.26053197e-01 -2.61297620e-01 -2.16236975e-01  3.44080560e-01 -1.21230621e-01 -6.02714323e-02 -2.19189671e-01  2.11470671e-01 -1.28634841e-01  5.76234521e-01 -5.50760430e-02  1.97333044e-01  2.05766455e-01  2.36621450e-02  3.70370109e-03  1.85197343e-04
        0.500     2.45944241e-01  7.99604140e-02 -3.58449873e-01  6.40998105e-02 -3.41792254e-01  2.24967546e-02  3.88717982e-01  1.77122204e-01 -2.55990758e-01 -6.44303562e-03 -3.75390123e-01 -7.62061467e-02 -5.02002036e-01  1.83662355e-02 -1.76351452e-01 -6.86345487e-02  1.54233464e-02  5.41822808e-03  1.27868850e-03
        0.750     2.25758567e-01  1.91381035e-01 -3.35176303e-01 -2.16152634e-01 -1.65178955e-01  4.23011572e-01 -1.44255462e-01  1.87567292e-01  1.49360082e-01  5.30105301e-01  4.17962321e-02  3.26784764e-01  2.74609836e-01  5.58724755e-02  4.42737636e-03  1.11352503e-02  2.44045339e-02  6.82889258e-04  1.98665379e-04
        1.000     2.11097169e-01  2.59405650e-01 -2.43643585e-01 -3.25745719e-01  7.63484286e-02  3.30279571e-01 -2.20001696e-01 -1.17395307e-01  2.71296591e-01 -4.38774328e-01  1.48165306e-01 -4.84545679e-01 -1.43691434e-01 -3.89147467e-02  8.93381613e-03 -2.05549231e-02 -6.11584426e-03  3.80246485e-03 -3.89229001e-04
        1.500     1.88387437e-01  3.29799741e-01 -9.46692612e-02 -2.73646379e-01  3.56651426e-01 -1.53161130e-01 -6.82050970e-04 -3.29897663e-01 -2.67361750e-01 -2.79382156e-01 -2.63740501e-01  5.28987613e-01  7.03281456e-02 -8.35258439e-02 -2.54953444e-02  2.71139319e-02  1.26194670e-02  3.85357399e-03 -8.09502038e-04
        2.000     1.76395533e-01  3.57332294e-01  5.54387509e-02 -1.55161958e-01  3.54513035e-01 -3.43041979e-01  1.61895880e-01 -2.75355961e-02 -2.07859409e-01  5.06833313e-01  2.05450556e-01 -4.13916086e-01 -4.07497251e-02  1.68472841e-01 -2.35465742e-03 -5.88317617e-03 -3.34222371e-03  1.97868686e-03  1.70392027e-03
        3.000     1.65469018e-01  3.60040620e-01  2.60392170e-01  6.69803672e-02  5.72701976e-02 -2.20913199e-01  1.81062550e-01  5.19913777e-01  4.62086625e-01 -1.04489885e-01 -2.19632037e-02  1.19011799e-01 -4.29988165e-03 -4.17545575e-01 -4.01081046e-02  2.00607204e-02 -5.26025220e-03 -5.80167364e-03  4.58230034e-04
        4.000     1.59580892e-01  3.47927160e-01  3.48346295e-01  2.39394141e-01 -1.57211928e-01  9.28779109e-02 -5.01602104e-03  1.69759688e-02  1.09978222e-01 -1.82603154e-01 -1.21233490e-01  7.11306931e-02  6.20582734e-02  7.50450107e-01  7.85420661e-02 -5.21102089e-02  7.51936647e-03 -1.88268150e-03 -1.85190445e-03
        5.000     1.48832921e-01  3.32847695e-01  3.65098052e-01  3.31227695e-01 -2.81334582e-01  2.83334016e-01 -1.82848345e-01 -3.25817997e-01 -3.10946154e-01  1.28556114e-01  8.37173663e-02 -7.03828761e-02 -4.61924126e-02 -4.41499964e-01 -4.05906261e-02  3.37161618e-02  1.16291918e-03  4.01605033e-03  5.05570345e-04
    """,
    )

    VARIANCE_SCALE_FACTOR = np.array(
        [
            0.63984174,
            0.84627714,
            0.90453306,
            0.93340282,
            0.95015459,
            0.96046156,
            0.96797214,
            0.97387821,
            0.97929703,
            0.98334580,
            0.98663949,
            0.98968140,
            0.99236758,
            0.99479044,
            0.99692908,
            0.99875974,
            0.99976488,
            0.99996981,
            1.0,
        ]
    )

    MODEL_VARIO = {
        1: {"Cn": 2.5, "C1": 4.52, "A1": 15.0, "C2": 6.78, "A2": 250.0, "type": "iso nest"},
        2: {"Cn": 0.5, "C1": 1.40, "A1": 10.0, "C2": 2.60, "A2": 160.0, "type": "iso nest"},
        3: {"Cn": 0.15, "C1": 0.42, "A1": 15.0, "C2": 0.63, "A2": 160.0, "type": "iso nest"},
        4: {"Cn": 0.15, "C1": 0.225, "A1": 10.0, "C2": 0.225, "A2": 120.0, "type": "iso nest"},
        5: {"Cn": 0.31432187, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        6: {"Cn": 0.19074954, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        7: {"Cn": 0.13784676, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        8: {"Cn": 0.11128384, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        9: {"Cn": 0.09649928, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        10: {"Cn": 0.0717368, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        11: {"Cn": 0.06481622, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        12: {"Cn": 0.05407664, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        13: {"Cn": 0.05118875, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        14: {"Cn": 0.04331642, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        15: {"Cn": 0.04139805, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        16: {"Cn": 0.03466367, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        17: {"Cn": 0.01879699, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        18: {"Cn": 0.00285694, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        19: {"Cn": 0.00036065, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
    }

    def __init__(self, **kwargs):
        """
        :param npcs: Number of principal components to be used
            (must be between 5 and 19)
        """

        super().__init__(**kwargs)
        self.npcs = int(kwargs.get("num_pcs", 5))
        assert (self.npcs >= 5) and (self.npcs <= 19), (
            "Number of principal components must be between 5 and 19 "
            f"({self.npcs} given)"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.npcs} Principal Components)"

    def get_correlation_model(self, distances: np.ndarray, imts: List):
        """Correlation model is not relevant in this context"""
        pass

    def get_lower_triangle_correlation_matrix(
            self, sites: SiteCollection, imts: List):
        """In this case the lower triangle correlation matrix has a different
        interpretation as here it has the dimension
        [num_sites, num_sites, num principal components]
        """
        # Get the distance matrix for the sites
        distance_matrix = sites.mesh.get_distance_matrix()
        model_vario = deepcopy(self.MODEL_VARIO)
        if self.npcs < 19:
            # If less than 19 principal components are used then scale down
            # the variance
            scale_factor = self.VARIANCE_SCALE_FACTOR[self.npcs - 1]
            for i in range(1, 20):
                model_vario[i]["Cn"] /= scale_factor
                if model_vario[i]["type"] != "nug":
                    model_vario[i]["C1"] /= scale_factor
                    model_vario[i]["C2"] /= scale_factor
        # Build the covariance matrices
        n_y, n_x = distance_matrix.shape
        self.cache["corma"] = np.empty([n_y, n_x, self.npcs])
        for i in range(self.npcs):
            var_model = model_vario[i + 1]
            if var_model["type"] == "nug":
                cov = get_nugget_cov(var_model, distance_matrix)
            else:
                cov = get_isotropic_nested_cov(var_model, distance_matrix)
            # Cholesky decomposition
            self.cache["corma"][:, :, i] = np.linalg.cholesky(cov)
        return

    def apply_correlation(
            self, sites: SiteCollection, imts: List, residuals: np.ndarray):
        """Apply the correlation models to the arrays on simulated residuals"""
        # Get the required PCA coefficients for the corresponding period
        pca_coeffs = {}
        for imt in imts:
            if str(imt) == "PGV":
                raise ValueError(
                    f"Correlation model {str(self)} not supported for PGV")
            pca_coeff = self.PCA_COEFFS[imt]
            pca_coeffs[imt] = np.array(
                [[pca_coeff["{:g}".format(i + 1)] for i in range(self.npcs)]]
            ).T
        nimts = len(imts)
        if not self.cache["corma"]:
            # Get the lower covariance matrices
            logging.info("Building lower triangle correlation matrices")
            self.get_lower_triangle_correlation_matrix(sites, imts)
            logging.info("Lower triangle correlation matrices built.")
        nlocs, nsims, _ = residuals.shape
        # Get simulated PCA matrices for each realisation of residuals
        logging.info("Generating spatially cross-correlated residuals")
        sim_pcas = np.empty([nlocs, nsims, self.npcs])
        for i in range(self.npcs):
            logging.info(
                "Processing principal component %g of %g" % (i + 1, self.npcs))
            for j in range(nsims):
                res = residuals[:, [j], [i]]
                sim_pcas[:, j, i] = (self.cache["corma"][:, :, i] @ res)[:, 0]

        sim_results = np.zeros([nimts, nlocs, nsims])
        for i, imt in enumerate(imts):
            for j in range(nsims):
                sim_results[i, :, j] = (
                    sim_pcas[:, j, :] @ pca_coeffs[imt])[:, 0]
        logging.info("Principal component processing completed.")
        return sim_results

    def sample(
        self,
        num_realizations: int,
        sites: SiteCollection,
        imts: List,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if not rng:
            rng = np.random.default_rng()
        uncorrelated_residuals = rng.normal(
            0.0, 1.0, [len(sites), num_realizations, self.npcs])
        return self.apply_correlation(sites, imts, uncorrelated_residuals)


class MonteiroEtAlGlobal2026CorrelationModel(BaseSpatialCrossCorrelationModel):
    """
    Implements the spatial cross-correlation model (global model) of
    Monteiro V.A., (2026) based on
    principal component analysis and geostatistics.

    Monteiro, V.A, Aristeidou, S. and O'Reilly, G.J. (2026). Spatial
    cross-correlation models for next-generation amplitude and cumulative
    intensity measures. Earthquake Spectra

    Attributes:
        npcs: Number of principal components to be used
        (must be between 3 and 22)
    """

    IMs = np.array(
        [
            'SA(0.1)',
            'SA(0.5)',
            'SA(1.0)',
            'SA(2.0)',
            'SA(3.0)',
            'Sa_avg2(0.1)',
            'Sa_avg2(0.5)',
            'Sa_avg2(1.0)',
            'Sa_avg2(2.0)',
            'Sa_avg2(3.0)',
            'Sa_avg3(0.1)',
            'Sa_avg3(0.5)',
            'Sa_avg3(1.0)',
            'Sa_avg3(2.0)',
            'Sa_avg3(3.0)',
            'FIV3(0.1)',
            'FIV3(0.5)',
            'FIV3(1.0)',
            'FIV3(2.0)',
            'FIV3(3.0)',
            'PGA',
            'PGV',
        ]
    )

    PCA_COEFFS = CoeffsTable(
        sa_damping=5,
        table="""\
        imt                       1            2            3            4            5            6            7            8            9           10           11           12           13           14           15           16           17           18           19           20           21           22
        SA(0.1)         0.135925640  0.425895839 -0.304606915  0.215062751	0.090762964	 0.214068375 -0.019522902  0.365363877	0.609631498	-0.110867056 -0.151692723 -0.027955892 -0.088830512	-0.042040118  0.070210847 -0.116691381 -0.027943943	 0.028561306 -0.107261362  0.086681025	0.074181340	 0.105391963
        SA(0.5)         0.183322706	 0.186191781  0.297506932 -0.469258899 -0.465951413	-0.296623665 -0.058762200  0.291364610	0.252421534	 0.165038896  0.166516239 -0.111470366  0.244123792	-0.011611688 -0.135229415  0.052474300 -0.010159638	 0.033378241 -0.081522994 -0.083252688	0.006655717	 0.053387842
        SA(1.0)         0.211835488 -0.016367706  0.397640891  0.133421264	0.126035527	 0.535801932  0.228683847 -0.293643500	0.204983605	 0.409680264  0.085597792 -0.132990641	0.186001950	 0.054228931 -0.209246130 -0.053197205 -0.044143152	 0.069113205 -0.027967290  0.030732392 -0.057380220	 0.081739653
        SA(2.0)         0.221822560	-0.174642377 -0.001307827  0.504314050 -0.087984181	-0.366669062 -0.491985706 -0.077535462	0.160547688	 0.319657007  0.019017470  0.041756772 -0.148232733	 0.171287993 -0.211936594 -0.135083432  0.116850905	-0.039240380  0.007375816 -0.065817510 -0.071982402	-0.080068339
        SA(3.0)         0.216286415	-0.211744610 -0.318130387  0.146790669 -0.351890882	-0.142898418  0.684447313 -0.023099142	0.030584922	-0.040101402 -0.072329408 -0.041461491	0.032764186	 0.287496964 -0.096049513 -0.136181568  0.104706389	-0.166786546  0.020909404 -0.090005721	0.063733406	 0.002583636
        Sa_avg2(0.1)    0.147704017	 0.419328798 -0.230014625  0.119174270	0.025833476	 0.063898394 -0.021201818  0.011270772 -0.173294313	 0.085624225 -0.002614084  0.084677461	0.259770272	 0.045567760 -0.228277887  0.481758931 -0.053478440	-0.086368953  0.450393104 -0.226966094 -0.074011094	-0.247885015
        Sa_avg2(0.5)    0.205560782	 0.211019175  0.224509564 -0.157223773 -0.172781347	 0.068794972  0.027545953 -0.016193682 -0.093532941	 0.083069815 -0.229579044  0.184158618 -0.373222220	 0.084275560  0.161301421 -0.190036755	0.290250964	 0.028369349  0.498256908  0.392512836 -0.019223462	-0.062762642
        Sa_avg2(1.0)    0.232906094	 0.028446169  0.227469062  0.129062179 -0.066973735	 0.025780410 -0.147493482 -0.067110848 -0.134906022	-0.159388968 -0.187964832 -0.187526482 -0.060288095	 0.010960500  0.150321577  0.143106630 -0.126397473	-0.334735354  0.107430839 -0.228614814	0.486024030	 0.506118741
        Sa_avg2(2.0)    0.242847633	-0.122874441 -0.045336653  0.105405583 -0.180115480	-0.028342355  0.067796941 -0.059091063 -0.009189205	-0.108337356 -0.059576622 -0.088587410 -0.193206107	-0.288101183  0.108232277  0.115096130 -0.171974694	 0.673750743  0.161139878 -0.261987392 -0.276069554	 0.220418058
        Sa_avg2(3.0)    0.235088181	-0.174112632 -0.208723859 -0.127705401 -0.122843848	 0.157208333 -0.120965285 -0.118316643	0.061630742	-0.116613381  0.091032849  0.066321052 -0.047503336	-0.352047894 -0.132652447  0.203055994	0.184548353	-0.445420791 -0.071255945  0.213661762 -0.458622322	 0.278354751
        Sa_avg3(0.1)    0.154169146	 0.403140216 -0.157418772  0.023244310 -0.074720701	-0.011534520 -0.046036991 -0.085095471 -0.498324925	 0.147770209 -0.217234584  0.190338668  0.186037411	-0.062976864 -0.141862301 -0.320459310	0.041340672	 0.135483114 -0.422434468  0.030055061 -0.040688145	 0.239288612
        Sa_avg3(0.5)    0.222262342	 0.118286784  0.244865139  0.008291649 -0.064748255	 0.135375204 -0.020436624 -0.014214755 -0.061447133	-0.195966760 -0.194755176 -0.162949209 -0.15442191	-0.003028258  0.244614559 -0.109932070	0.010826480 -0.241584206 -0.280339819 -0.408761888 -0.306713991	-0.502766158
        Sa_avg3(1.0)    0.242951743	-0.046412545  0.066748683  0.190233898 -0.151323153	-0.081518982  0.001332679 -0.037190805 -0.070122153	-0.069259813 -0.118034033 -0.076186718  0.113068211	 0.098191125  0.187008287  0.412549251 -0.315731094	 0.095492110 -0.280914157  0.605941336	0.064778650	-0.226103595
        Sa_avg3(2.0)    0.237248768	-0.154234894 -0.205108842 -0.134118480 -0.129924219	 0.167259367 -0.109659948 -0.164464841  0.025823657	-0.019486942  0.058369302  0.027229155 -0.018419170 -0.485551630 -0.173867654 -0.134818788	0.059287675	 0.079494160 -0.008579490 -0.013893918	0.562252721	-0.400009232
        Sa_avg3(3.0)    0.219848464	-0.176862315 -0.286890567 -0.356754602 -0.024634216	 0.252354379 -0.346679081 -0.213386608	0.106068753	-0.121269744 -0.006371576  0.141032841	0.216395053	 0.548463329  0.208909414 -0.083820137 -0.075687469	 0.138572169  0.037394192 -0.091951988 -0.017511413	 0.038858075
        FIV3(0.1)       0.228923987	 0.010300995  0.232469904 -0.029120446	0.195023193	-0.133762317  0.103626729  0.007055554	0.061032524	-0.362675536  0.130076252  0.502880397 -0.106485633	 0.028505024 -0.347141087 -0.186374930 -0.492510408	-0.081091062  0.053279783  0.007791766 -0.035732980	-0.010143409
        FIV3(0.5)       0.231549529	-0.079028972  0.214156250  0.106398076	0.196913760	 0.037884632  0.056187977  0.213684841 -0.044978131	-0.268531093  0.161413910  0.271998819	0.067292398	 0.117096864 -0.033087385  0.282985918	0.632949827	 0.225627478 -0.209363579 -0.078511395  0.133670772	 0.030715028
        FIV3(1.0)       0.234055192	-0.143068708  0.044134728  0.178174413	0.170016118	-0.065533550 -0.042980331  0.280826435 -0.145154146	-0.141630622  0.138341875 -0.204991522	0.549674382	-0.180753987  0.231598241 -0.395824102	0.018754492	-0.045713111  0.305798504  0.140044224 -0.116285362	-0.025267097
        FIV3(2.0)       0.230112173	-0.157553595 -0.113490261 -0.089557491	0.185838467	-0.002733499  0.127164162  0.300583508 -0.081938915	 0.556192813  0.126726282  0.356235441 -0.136120756	-0.107598358  0.429541096  0.101875777 -0.172924367	-0.131946559 -0.080054545 -0.140770137  0.049480523	 0.015997258
        FIV3(3.0)       0.223534065	-0.149771535 -0.119896544 -0.225231821	0.279322296	 0.098460754 -0.051361149  0.402673436 -0.271745958	 0.054887155 -0.054640175 -0.461622418 -0.314150456	 0.198619064 -0.403903742  0.008987361 -0.071696476	 0.053353502 -0.024229093  0.082586104 -0.014334545	-0.009941565
        PGA             0.168501568	 0.362160761 -0.121224871  0.007123073	0.089290615	-0.138881354  0.060204080 -0.292995192 -0.071898471	-0.058992196  0.722159436 -0.249721220 -0.245712199	 0.087409427  0.189317198 -0.062756133	0.021245565	-0.001256152 -0.047401856  0.061373806  0.039507412	 0.024958721
        PGV             0.213470506	 0.052739442 -0.036445378 -0.268495629	0.517932378	-0.468898279  0.135030388 -0.349477293	0.231423352	 0.064842413 -0.362337983 -0.129135597	0.104233036	-0.104265433  0.035913972  0.069090490	0.126649676	 0.009687264 -0.004816915  0.001118124 -0.007737344	 0.000237981
    """,
    )  # noqa: E501

    VARIANCE_SCALE_FACTOR = np.array(
        [
            0.771784785,
            0.925420389,
            0.959863331,
            0.971398217,
            0.980166443,
            0.985712880,
            0.989848342,
            0.992480027,
            0.994338550,
            0.995605123,
            0.996661862,
            0.997476859,
            0.998088659,
            0.998647219,
            0.999069328,
            0.999324596,
            0.999524952,
            0.999662491,
            0.999784661,
            0.999883599,
            0.999953687,
            1
        ]
    )

    MODEL_VARIO = {
        1: {"Cn": 8.14481289, "C1": 13.20035514, "A1": 94.14906432, "C2": 0.00, "A2": 216.8645443, "type": "iso nest"},
        2: {"Cn": 0.953752621, "C1": 2.257911792, "A1": 61.67331411, "C2": 1.429126771, "A2": 226.4852379, "type": "iso nest"},
        3: {"Cn": 0.286321863, "C1": 0.413021836, "A1": 46.64523871, "C2": 0.638822635, "A2": 227.7251389, "type": "iso nest"},
        4: {"Cn": 0.111971649, "C1": 0.093559985, "A1": 28.43299213, "C2": 0.244837227, "A2": 228.1269782, "type": "iso nest"},
        5: {"Cn": 0.095674817, "C1": 0.122092004, "A1": 31.26680244, "C2": 0.065942052, "A2": 228.0093191, "type": "iso nest"},
        6: {"Cn": 0.06665654, "C1": 0.070614382, "A1": 28.0277504, "C2": 0.054436852, "A2": 228.023073, "type": "iso nest"},
        7: {"Cn": 0.049370678, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        8: {"Cn": 0.037466934, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        9: {"Cn": 0.030873087, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        10: {"Cn": 0.022642474, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        11: {"Cn": 0.017726169, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        12: {"Cn": 0.01420243, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        13: {"Cn": 0.01151974, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        14: {"Cn": 0.006642143, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        15: {"Cn": 0.00818584, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        16: {"Cn": 0.004073281, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        17: {"Cn": 0.004029638, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        18: {"Cn": 0.002213756, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        19: {"Cn": 0.002462062, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        20: {"Cn": 0.001954815, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        21: {"Cn": 0.001336528, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        22: {"Cn": 0.000919364, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
    }

    def __init__(self, **kwargs):
        """
        Args:
            num_pcs = Number of principal components to be used
        """

        super().__init__(**kwargs)
        self.npcs = int(kwargs.get("num_pcs", 3))
        assert (self.npcs >= 3) and (self.npcs <= 22), (
            "Number of principal components must be between 3 and 22 "
            "(%g given)" % self.npcs
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.npcs} Principal Components)"

    def get_correlation_model(self, distances: np.ndarray, imts: List):
        """Correlation model is not relevant in this context"""
        pass

    def get_lower_triangle_correlation_matrix(
            self, sites: SiteCollection, imts: List):
        """In this case the lower triangle correlation matrix has a different
        interpretation as here it has the dimension
        [num_sites, num_sites, num principal components]
        """
        # Get the distance matrix for the sites
        distance_matrix = sites.mesh.get_distance_matrix()
        model_vario = deepcopy(self.MODEL_VARIO)
        if self.npcs < 22:
            # If less than 22 principal components are used then scale down
            # the variance
            scale_factor = self.VARIANCE_SCALE_FACTOR[self.npcs - 1]
            for i in range(1, 23):
                model_vario[i]["Cn"] /= scale_factor
                if model_vario[i]["type"] != "nug":
                    model_vario[i]["C1"] /= scale_factor
                    model_vario[i]["C2"] /= scale_factor
        # Build the covariance matrices
        n_y, n_x = distance_matrix.shape
        self.cache["corma"] = np.empty([n_y, n_x, self.npcs])
        for i in range(self.npcs):
            var_model = model_vario[i + 1]
            if var_model["type"] == "nug":
                cov = get_nugget_cov(var_model, distance_matrix)
            else:
                cov = get_isotropic_nested_cov(var_model, distance_matrix)
            self.cache["corma"][:, :, i] = np.linalg.cholesky(cov)
        return

    def apply_correlation(
            self, sites: SiteCollection, imts: List, residuals: np.ndarray):
        """Apply the correlation models to the arrays on simulated residuals"""
        # Get the required PCA coefficients for the corresponding period
        pca_coeffs = {}
        for imt in imts:
            pca_coeff = self.PCA_COEFFS[imt]
            pca_coeffs[imt] = np.array(
                [[pca_coeff["{:g}".format(i + 1)] for i in range(self.npcs)]]
            ).T
        nimts = len(imts)
        if not self.cache["corma"]:
            # Get the lower covariance matrices
            logging.info("Building lower triangle correlation matrices")
            self.get_lower_triangle_correlation_matrix(sites, imts)
            logging.info("Lower triangle correlation matrices are built")
        nlocs, nsims, _ = residuals.shape
        # Get simulated PCA matrices for each realisation of residuals
        logging.info("Generating spatially cross-correlated residuals")
        sim_pcas = np.empty([nlocs, nsims, self.npcs])
        for i in range(self.npcs):
            logging.info(
                "Processing principal component %g of %g" % (i + 1, self.npcs))
            for j in range(nsims):
                res = residuals[:, [j], [i]]
                sim_pcas[:, j, i] = (self.cache["corma"][:, :, i] @ res)[:, 0]

        sim_results = np.zeros([nimts, nlocs, nsims])
        for i, imt in enumerate(imts):
            for j in range(nsims):
                sim_results[i, :, j] = (
                    sim_pcas[:, j, :] @ pca_coeffs[imt])[:, 0]
        logging.info(
            "Correlation models to the arrays on simulated residuals applied")
        return sim_results

    def sample(
        self,
        num_realizations: int,
        sites: SiteCollection,
        imts: List,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """ """
        if not rng:
            rng = np.random.default_rng()
        uncorrelated_residuals = rng.normal(
            0.0, 1.0, [len(sites), num_realizations, self.npcs])
        return self.apply_correlation(sites, imts, uncorrelated_residuals)


class DuNing2021CorrelationModel(BaseSpatialCrossCorrelationModel):
    """
    Implements the spatial cross-correlation model of Du and Ning, (2021)
    based on principal component analysis and geostatistics.

    Du, W., & Ning, C.-L. (2021). Modeling spatial cross-correlation of
    multiple ground motion intensity measures (SAs, PGA, PGV, Ia, CAV, and
    significant durations) based on principal component and geostatistical
    analyses. Earthquake Spectra, 37(1), 486–504.
    https://doi.org/10.1177/8755293020952442

    Attributes:
        npcs: Number of principal components to be used (must be or 7 or 23)
    """
    IMs = np.array(
        [
            'SA(0.01)',
            'SA(0.05)',
            'SA(0.075)',
            'SA(0.1)',
            'SA(0.2)',
            'SA(0.3)',
            'SA(0.4)',
            'SA(0.5)',
            'SA(0.75)',
            'SA(1.0)',
            'SA(1.5)',
            'SA(2.0)',
            'SA(3.0)',
            'SA(4.0)',
            'SA(5.0)',
            'SA(7.5)',
            'SA(10)',
            'PGA',
            'PGV',
            'DS575',
            'DS595'
        ]
    )

    PCA_COEFFS = CoeffsTable(
        sa_damping=5,
        table="""\
        imt         1        2      3       4        5       6       7      8       9       10.    11       12      13      14      15     16.      17      18      19      20      21      22      23
        SA(0.01)    0.28    -0.15   0.07    0.04	-0.05	-0.10	-0.08	0.11	0.00	-0.25	0.04	-0.19	-0.07	-0.03	0.03	0.04	0.00	-0.05	-0.21	0.03	-0.15	-0.43	-0.71
        SA(0.05)    0.24 	-0.19	0.23	0.04	-0.18	-0.10	0.06	-0.10	-0.14	-0.16	0.21	-0.23	-0.09	0.07	-0.17	0.05	0.12	0.2 	0.3 	0.07	-0.48	0.48	0.00
        SA(0.075)   0.21	-0.21	0.29	0.08	-0.17	0.06	0.34	-0.24	-0.16	0.08	0.11	0.01	-0.02	0.06	-0.05	0.00	0.02	0.07	0.4 	-0.04	0.56	-0.29	0.00
        SA(0.1)     0.2	    -0.22	0.27	0.08	-0.09	0.15	0.45	-0.11	-0.06	0.21	-0.22	0.26	0.17	-0.08	0.25	-0.05	-0.13	-0.25	-0.39	-0.08	-0.25	0.16	0.00
        SA(0.2)     0.22	-0.20	0.09	0.04	0.23	0.3	    0.11	0.53	0.47	-0.05	-0.31	0.05	-0.20	-0.05	-0.05	-0.08	0.05	0.2 	0.2 	-0.01	0.06	0.1 	0.00
        SA(0.3)     0.24	-0.14	-0.11	0.02	0.39	0.28	-0.11	0.19	-0.06	0.15	0.53	0.08	0.33	0.16	-0.05	0.2 	0.03	-0.36	0.12	0.01	-0.03	0.02	0.00
        SA(0.4)     0.24	-0.09	-0.19	-0.05	0.37	0.2	    -0.13	-0.19	-0.35	0.1 	0.09	0.00	-0.15	-0.01	0.18	-0.35	-0.15	0.53	-0.18	-0.08	0.03	0.02	0.00
        SA(0.5)     0.24	-0.04	-0.25	-0.11	0.31	0.02	-0.01	-0.35	-0.19	-0.09	-0.47	0.03	-0.23	-0.17	-0.28	0.24	0.14	-0.33	0.13	0.04	-0.03	0.02	0.00
        SA(0.75)    0.23	0.05	-0.30	-0.20	0.08	-0.19	0.22	-0.26	0.37	-0.18	-0.12	-0.13	0.42	0.34	0.33	-0.02	0.15	0.13	0.08	0.11	0.02	0.02	0.00
        SA(1.0)     0.22	0.13	-0.27	-0.27	-0.10	-0.17	0.32	-0.05	0.31	0.00	0.37	0.21	-0.17	-0.24	-0.40	-0.20	-0.25	-0.02	-0.10	-0.07	-0.03	-0.01	0.00
        SA(1.5)     0.19	0.22	-0.20	-0.26	-0.23	0.02	0.19	0.28	-0.18	0.32	0.06	-0.09	-0.26	-0.07	0.3	    0.51	0.15	0.15	-0.05	0.11	0.03	0.02	0.00
        SA(2.0)     0.19	0.27	-0.10	-0.17	-0.27	0.2	    0.06	0.3     -0.34	0.06	-0.26	-0.25	0.28	0.23	-0.33	-0.38	-0.02	-0.11	0.01	-0.03	0.00	-0.02	0.00
        SA(3.0)     0.17	0.3 	0.06	0.00    -0.25	0.38	-0.20	-0.10	-0.03	-0.44	-0.02	0.54	0.06	0.13	0.01	0.24	-0.16	0.17	0.04	0.03	-0.04	-0.03	0.00
        SA(4.0)     0.16	0.32	0.12	0.11	-0.11	0.32	-0.14	-0.24	0.24	0.11	0.16	-0.09	-0.11	-0.20	0.08	-0.26	0.63	-0.12	-0.09	-0.08	-0.01	-0.01	0.00
        SA(5.0)     0.15	0.32	0.15	0.24	0.06	0.17	-0.08	-0.23	0.27	0.27	-0.05	-0.40	-0.09	0.1 	-0.04	0.14	-0.54	-0.04	0.02	0.24	-0.03	0.00	0.00
        SA(7.5)     0.12	0.31	0.08	0.37	0.23	-0.25	0.16	0.06	-0.03	0.03	-0.04	-0.01	0.13	0.12	-0.25	0.28	0.12	0.23	-0.18	-0.57	0.02	0.00	0.00
        SA(10)      0.12	0.3 	0.01	0.41	0.2	    -0.30	0.18	0.17	-0.19	-0.03	0.02	0.26	0.01	-0.09	0.05	-0.21	0.11	0.00	0.08	0.6 	-0.01	-0.02	0.00
        PGA         0.28	-0.15	0.07	0.04	-0.05	-0.10	-0.08	0.11	0   	-0.25	0.05	-0.19	-0.07	-0.03	0.03	0.04	0.00	-0.05	-0.21	0.03	-0.14	-0.43	0.71
        PGV         0.26	0.13	-0.08	0.09	-0.10	-0.29	-0.23	0.11	-0.03	0.01	-0.04	0.09	-0.16	-0.04	0.44	-0.21	-0.23	-0.30	0.39	-0.41	-0.02	0.09	0.00
        Ia          0.23	-0.12	0.19	-0.19	-0.10	-0.30	-0.43	-0.05	0.13	0.52	-0.16	0.3 	0.19	0.03	-0.19	0.02	0.09	0.19	0.04	0.11	-0.11	-0.15	0.00
        CAV         0.26	0.01	0.25	-0.17	0.02	-0.19	-0.23	0.07	-0.01	-0.21	0.03	-0.10	0.07	-0.12	-0.01	0.04	-0.03	-0.10	-0.35	0.1 	0.55	0.47	0.00
        RSD575	    -0.09	0.21	0.41	-0.40	0.29	-0.09	0.10 	0.00	-0.02	-0.01	0.06	0.11	-0.39	0.54	0.04	-0.10	0.06	-0.15	-0.06	0.00	-0.08	-0.07   0.00
        RSD595	    -0.07	0.27	0.36	-0.38	0.27	0.02	0.09	0.03	-0.07	-0.10	-0.01	-0.12	0.34	-0.54	0.11	0.04	-0.09	0.1 	0.24	-0.06	-0.15	-0.14	0.00
    """,
    )  # noqa: E501

    MODEL_VARIO = {
        1: {"Cn": 1.03, "C1": 0.88, "A1": 15.0, "C2": 10.11, "A2": 200.0, "type": "iso nest"},
        2: {"Cn": 0.36, "C1": 1.76, "A1": 25.0, "C2": 2.61, "A2": 150.0, "type": "iso nest"},
        3: {"Cn": 0.13, "C1": 0.37, "A1": 25.0, "C2": 1.75, "A2": 200.0, "type": "iso nest"},
        4: {"Cn": 0.09, "C1": 0.26, "A1": 20.0, "C2": 1.11, "A2": 150.0, "type": "iso nest"},
        5: {"Cn": 0.10, "C1": 0.32, "A1": 15.0, "C2": 0.45, "A2": 150.0, "type": "iso nest"},
        6: {"Cn": 0.11, "C1": 0.13, "A1": 25.0, "C2": 0.35, "A2": 250.0, "type": "iso nest"},
        7: {"Cn": 0.06, "C1": 0.16, "A1": 25.0, "C2": 0.25, "A2": 250.0, "type": "iso nest"},
        8: {"Cn": 0.10, "C1": 0.13, "A1": 20.0, "C2": 0.16, "A2": 200.0, "type": "iso nest"},
        9: {"Cn": 0.11, "C1": 0.07, "A1": 25.0, "C2": 0.08, "A2": 120.0, "type": "iso nest"},
        10: {"Cn": 0.13, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        11: {"Cn": 0.16, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        12: {"Cn": 0.14, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        13: {"Cn": 0.15, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        14: {"Cn": 0.14, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        15: {"Cn": 0.12, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        16: {"Cn": 0.13, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        17: {"Cn": 0.12, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        18: {"Cn": 0.12, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        19: {"Cn": 0.09, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        20: {"Cn": 0.09, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        21: {"Cn": 0.06, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        22: {"Cn": 0.04, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
        23: {"Cn": 0.00, "C1": 0.0, "A1": 0.0, "C2": 0.0, "A2": 0.0, "type": "nug"},
    }

    def __init__(self, **kwargs):
        """
        Args:
            num_pcs = Number of principal components to be used
        """

        super().__init__(**kwargs)
        self.npcs = int(kwargs.get("num_pcs", 7))
        if self.npcs not in (7, 23):
            raise ValueError(
                "Number of principal components must be either 7 or 23 "
                "(%g given)" % self.npcs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.npcs} Principal Components)"

    def get_correlation_model(self, distances: np.ndarray, imts: List):
        """Correlation model is not relevant in this context"""
        pass

    def get_lower_triangle_correlation_matrix(
            self, sites: SiteCollection, imts: List):
        """In this case the lower triangle correlation matrix has a different
        interpretation as here it has the dimension
        [num_sites, num_sites, num principal components]
        """
        # Get the distance matrix for the sites
        distance_matrix = sites.mesh.get_distance_matrix()
        model_vario = deepcopy(self.MODEL_VARIO)
        if self.npcs == 7:
            # If 7 principal components are used then scale down the variance
            # of 0.90
            scale_factor = 0.90
            for i in range(1, 24):
                model_vario[i]["Cn"] /= scale_factor
                if model_vario[i]["type"] != "nug":
                    model_vario[i]["C1"] /= scale_factor
                    model_vario[i]["C2"] /= scale_factor
        # Build the covariance matrices
        n_y, n_x = distance_matrix.shape
        self.cache["corma"] = np.empty([n_y, n_x, self.npcs])
        for i in range(self.npcs):
            var_model = model_vario[i + 1]
            if var_model["type"] == "nug":
                cov = get_nugget_cov(var_model, distance_matrix)
            else:
                cov = get_isotropic_nested_cov(var_model, distance_matrix)
            self.cache["corma"][:, :, i] = np.linalg.cholesky(cov)
        return

    def apply_correlation(
            self, sites: SiteCollection, imts: List, residuals: np.ndarray):
        """Apply the correlation models to the arrays on simulated residuals"""
        # Get the required PCA coefficients for the corresponding period
        pca_coeffs = {}
        for imt in imts:
            pca_coeff = self.PCA_COEFFS[imt]
            pca_coeffs[imt] = np.array(
                [[pca_coeff["{:g}".format(i + 1)] for i in range(self.npcs)]]
            ).T
        nimts = len(imts)
        if not self.cache["corma"]:
            # Get the lower covariance matrices
            logging.info("--- Building lower triangle correlation matrices")
            self.get_lower_triangle_correlation_matrix(sites, imts)
            logging.info("--- done!")
        nlocs, nsims, _ = residuals.shape
        # Get simulated PCA matrices for each realisation of residuals
        logging.info("--- Generating spatially cross-correlated residuals")
        sim_pcas = np.empty([nlocs, nsims, self.npcs])
        for i in range(self.npcs):
            logging.info(
                "Processing principal component %g of %g" % (i + 1, self.npcs))
            for j in range(nsims):
                res = residuals[:, [j], [i]]
                sim_pcas[:, j, i] = (self.cache["corma"][:, :, i] @ res)[:, 0]

        sim_results = np.zeros([nimts, nlocs, nsims])
        for i, imt in enumerate(imts):
            for j in range(nsims):
                sim_results[i, :, j] = (
                    sim_pcas[:, j, :] @ pca_coeffs[imt])[:, 0]
        logging.info("--- --- done!")
        return sim_results

    def sample(
        self,
        num_realizations: int,
        sites: SiteCollection,
        imts: List,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """ """
        if not rng:
            rng = np.random.default_rng()
        uncorrelated_residuals = rng.normal(
            0.0, 1.0, [len(sites), num_realizations, self.npcs])
        return self.apply_correlation(sites, imts, uncorrelated_residuals)


class MonteiroEtAlPairWise2026CorrelationModel(
        BaseSpatialCrossCorrelationModel):
    """
    Implements the spatial cross-correlation model (pairwise model) of
     Monteiro V.A., (2026) based on principal component analysis and
     geostatistics.

    Monteiro, V.A, Aristeidou, S. and O'Reilly, G.J. (2026). Spatial
    cross-correlation models for next-generation amplitude and cumulative
    intensity measures. Earthquake Spectra

    Note:
        This model should be applied to only two IMs. If you want to perform
        analysis with more than 2 IMs please use
        'MonteiroEtAlGlobalCorrelationModel'.

    Attributes:
        npcs: Number of principal components is fixed to 2 because it is a
        pairwise model
    """

    def __init__(self, **kwargs):
        """
        Args:
            npcs = Number of principal components to be used which are fixed
            hdf5_file = File that contains all pca coefficients and model
            parameters
        """
        self.npcs = 2  # fixed by your HDF5
        self.cache = {}
        # Load HDF5 data once

        hdf5_path = Path(__file__).parent / \
            'pairwisemodels_monteiroetal26.hdf5'

        self.data = {
            "model_parameters": {},
            "pca_coeff": {}
        }
        with h5py.File(hdf5_path, 'r') as f:

            # Load model parameters
            for pair_name in f['model_parameters'].keys():
                self.data['model_parameters'][pair_name] = {}
                for period_key in f['model_parameters'][pair_name].keys():
                    self.data['model_parameters'][pair_name][period_key] = (
                        f['model_parameters'][pair_name][period_key][:]
                    )
            # Load PCA coefficients
            for pair_name in f['pca_coeff'].keys():
                self.data['pca_coeff'][pair_name] = {}
                for period_key in f['pca_coeff'][pair_name].keys():
                    self.data['pca_coeff'][pair_name][period_key] = (
                        f['pca_coeff'][pair_name][period_key][:]
                    )

    def __repr__(self):
        return f"{self.__class__.__name__}(npcs={self.npcs})"

    def get_correlation_model(self, distances: np.ndarray, imts: List):
        """Correlation model is not relevant in this context"""
        pass

    @staticmethod
    def _interpolate_dfs(
        low_df: np.ndarray, high_df: np.ndarray, ratio: float
    ) -> np.ndarray:
        """
        Interpolates between two arrays handling 1-PC vs 2-PC cases.
        """
        if low_df.shape == (1, 1) and high_df.shape == (1, 1):
            return np.array([[low_df[0, 0] + (high_df[0, 0] - low_df[0, 0])
                              * ratio]], dtype=float)
        if low_df.shape == (1, 1):
            return high_df.astype(float)
        if high_df.shape == (1, 1):
            return low_df.astype(float)
        if low_df.shape[1] == 1 and high_df.shape[1] == 2:
            baseline = np.array([1.0, 1.0], dtype=float)
            low_vals = np.column_stack([low_df[:, 0], baseline[1:]])
            return low_vals + (high_df - low_vals) * ratio
        if high_df.shape[1] == 1 and low_df.shape[1] == 2:
            baseline = np.array([1.0, 1.0], dtype=float)
            high_vals = np.column_stack([high_df[:, 0], baseline[1:]])
            return low_df + (high_vals - low_df) * ratio
        return low_df + (high_df - low_df) * ratio

    @staticmethod
    def _extract_period(imt) -> float:
        """
        Extract period from an OpenQuake IMT object.
        Works for SA-type IMTs.
        """
        if not hasattr(imt, "period") or imt.period is None:
            raise ValueError(f"IMT {imt} does not have a period.")
        return imt.period

    def _get_pairwise_key(self, imt1, imt2) -> str:
        period1 = imt1.period
        period2 = imt2.period
        return f"{period1:.2f}_{period2:.2f}"

    def _get_pair_name(self, imt1, imt2):
        name1 = imt1.string.split("(")[0]
        name2 = imt2.string.split("(")[0]
        return f"{name1}-{name2}"

    def _resolve_pair_name(self, imt1, imt2):
        """
        Return a valid pair_name stored in the HDF5.
        """
        name1 = imt1.string.split("(")[0]
        name2 = imt2.string.split("(")[0]

        direct = f"{name1}-{name2}"
        reverse = f"{name2}-{name1}"

        available_pairs = self.data["model_parameters"].keys()

        if direct in available_pairs:
            return direct, False  # False = not reversed

        if reverse in available_pairs:
            return reverse, True  # True = reversed order

        raise ValueError(
            f"Cross-IM spatial correlation model not available for pair "
            f"'{name1}' and '{name2}'."
        )

    def get_model_parameters(self, imt1: str, imt2: str) -> np.ndarray:
        """
        Return model parameters for a pair of IMTs with interpolation
        if needed.
        """

        pair_name, reversed_order = self._resolve_pair_name(imt1, imt2)
        key = self._get_pairwise_key(imt1, imt2)

        cache_key = f"params_{pair_name}_{key}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        available = sorted(
            self.data["model_parameters"][pair_name].keys(),
            key=lambda x: float(x.split("_")[0])
        )

        if key in available:
            params = self.data["model_parameters"][pair_name][key]
        else:
            # Interpolation between nearest keys
            periods = np.array([float(k.split("_")[0]) for k in available])

            T = self._extract_period(imt1)
            lower_idx = np.searchsorted(periods, T, side='right') - 1
            upper_idx = lower_idx + 1

            key_low = available[max(lower_idx, 0)]
            key_high = available[min(upper_idx, len(available) - 1)]

            low_params = self.data["model_parameters"][pair_name][key_low]
            high_params = self.data["model_parameters"][pair_name][key_high]

            T_low = float(key_low.split("_")[0])
            T_high = float(key_high.split("_")[0])

            ratio = (T - T_low) / (T_high - T_low + 1e-12)

            params = self._interpolate_dfs(low_params, high_params, ratio)

        self.cache[cache_key] = params
        return params

    def get_pca_coeff(self, imt1: str, imt2: str) -> np.ndarray:
        """
        Return PCA coefficients for a pair of IMTs with interpolation
        if needed.
        """

        pair_name, reversed_order = self._resolve_pair_name(imt1, imt2)
        key = self._get_pairwise_key(imt1, imt2)

        cache_key = f"pca_{pair_name}_{key}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        available = sorted(
            self.data["pca_coeff"][pair_name].keys(),
            key=lambda x: float(x.split("_")[0])
        )

        if key in available:
            coeffs = self.data["pca_coeff"][pair_name][key]
        else:
            periods = np.array([float(k.split("_")[0]) for k in available])

            T = self._extract_period(imt1)
            lower_idx = np.searchsorted(periods, T, side='right') - 1
            upper_idx = lower_idx + 1

            key_low = available[max(lower_idx, 0)]
            key_high = available[min(upper_idx, len(available) - 1)]

            low_coeffs = self.data["pca_coeff"][pair_name][key_low]
            high_coeffs = self.data["pca_coeff"][pair_name][key_high]

            T_low = float(key_low.split("_")[0])
            T_high = float(key_high.split("_")[0])

            ratio = (T - T_low) / (T_high - T_low + 1e-12)
            coeffs = self._interpolate_dfs(low_coeffs, high_coeffs, ratio)

        if reversed_order:
            coeffs = coeffs[:, ::-1]
        self.cache[cache_key] = coeffs
        return coeffs

    def get_lower_triangle_correlation_matrix(self, sites, imts: list):
        """Compute lower-triangle correlation matrix for a pair of IMTs."""
        distance_matrix = sites.mesh.get_distance_matrix()
        n_y, n_x = distance_matrix.shape
        self.cache["corma"] = np.empty([n_y, n_x, self.npcs])

        imt1, imt2 = imts
        model_params = self.get_model_parameters(imt1, imt2)

        for i in range(self.npcs):
            var_model = {
                "Cn": model_params[i, 0],
                "C1": model_params[i, 1],
                "A1": model_params[i, 2],
                "C2": model_params[i, 3],
                "A2": model_params[i, 4],
                "type": "iso nest"
            }
            cov = get_isotropic_nested_cov(var_model, distance_matrix)
            self.cache["corma"][:, :, i] = np.linalg.cholesky(cov)

    def apply_correlation(self, sites, imts: List[str], residuals: np.ndarray):
        """Apply spatial cross-correlation to a list of IMTs."""
        imt1, imt2 = imts
        pca_coeffs = self.get_pca_coeff(imt1, imt2)
        self.get_lower_triangle_correlation_matrix(sites, imts)
        sim_pcas = np.einsum('ijk,jlk->ilk', self.cache["corma"], residuals)
        sim_results = sim_pcas @ pca_coeffs

        return sim_results

    def sample(
        self,
        num_realizations: int,
        sites,
        imts: List[str],
        rng: Optional[np.random.Generator] = None
    ):
        if rng is None:
            rng = np.random.default_rng()
        uncorrelated_residuals = rng.normal(
            0.0, 1.0, [len(sites), num_realizations, self.npcs])
        return self.apply_correlation(sites, imts, uncorrelated_residuals)
