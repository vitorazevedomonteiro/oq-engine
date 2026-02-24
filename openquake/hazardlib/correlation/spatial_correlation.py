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
"""
Module :mod:`openquake.hazardlib.correlation` defines correlation models for
spatially-distributed ground-shaking intensities.
"""
import abc
import logging
import numpy


class BaseCorrelationModel(metaclass=abc.ABCMeta):
    """
    Base class for spatial correlation models for spatially-distributed
    ground-shaking intensities.
    """

    def apply_correlation(self, sites, imt, residuals, stddev_intra=0):
        """
        Apply correlation to randomly sampled residuals.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` residuals were
            sampled for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        :param residuals:
            2d numpy array of sampled residuals, where first dimension
            represents sites (the length as ``sites`` parameter) and
            second one represents different realizations (samples).
        :param stddev_intra:
            Intra-event standard deviation array (phi). Different sites do
            not necessarily have the same intra-event standard deviation.
        :returns:
            Array of the same structure and semantics as ``residuals``
            but with correlations applied.

        NB: the correlation matrix is cached. It is computed only once
        per IMT for the complete site collection and then the portion
        corresponding to the sites is multiplied by the residuals.
        """
        # intra-event residual for a single relization is a product
        # of lower-triangle decomposed correlation matrix and vector
        # of N random numbers (where N is equal to number of sites).
        # we need to do that multiplication once per realization
        # with the same matrix and different vectors.
        try:
            corma = self.cache[imt]
        except KeyError:
            logging.info("--- Building lower triangle correlation matrix")
            corma = self.get_lower_triangle_correlation_matrix(
                sites.complete, imt)
            logging.info("--- --- done!")
            self.cache[imt] = corma
        # if N is the length of the complete site collection, then the
        # correlation matrix has shape (N, N) and the residuals (N, s),
        # where s is the number of samples
        N = len(sites.complete)
        n = len(sites)
        if n < N:  # filtered site collection
            res = numpy.zeros((N, residuals.shape[1]))
            res[sites.sids] = residuals
            return (corma @ res)[sites.sids, :]  # shape (n, s)
        else:  # complete site collection
            return corma @ residuals  # shape (N, s)


class JB2009CorrelationModel(BaseCorrelationModel):
    """
    "Correlation model for spatially distributed ground-motion intensities"
    by Nirmal Jayaram and Jack W. Baker. Published in Earthquake Engineering
    and Structural Dynamics 2009; 38, pages 1687-1708.

    :param vs30_clustering:
        Boolean value to indicate whether "Case 1" or "Case 2" from page 1700
        should be applied. ``True`` value means that Vs 30 values show or are
        expected to show clustering ("Case 2"), ``False`` means otherwise.
    """

    def __init__(self, vs30_clustering):
        self.vs30_clustering = vs30_clustering
        self.cache = {}  # imt -> correlation model

    def _get_correlation_matrix(self, sites, imt):
        return jbcorrelation(sites, imt, self.vs30_clustering)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`JB2009CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))


def jbcorrelation(sites_or_distances, imt, vs30_clustering=False):
    """
     Returns the Jayaram-Baker correlation model.

     :param sites_or_distances:
         SiteCollection instance o ristance matrix
     :param imt:
         Intensity Measure Type (PGA or SA)
     :param vs30_clustering:
         flag, defalt false
    """
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    # formulae are from page 1700
    period = 1.0 if imt.string == 'PGV' else imt.period
    if period < 1:
        if not vs30_clustering:
            # case 1, eq. (17)
            b = 8.5 + 17.2 * imt.period
        else:
            # case 2, eq. (18)
            b = 40.7 - 15.0 * imt.period
    else:
        # both cases, eq. (19)
        b = 22.0 + 3.7 * imt.period

    # eq. (20)
    return numpy.exp((- 3.0 / b) * distances)


class HM2018CorrelationModel(BaseCorrelationModel):
    """
    "Uncertainty in intraevent spatial correlation of elastic pseudo-
    acceleration spectral ordinates"
    by Pablo Heresi and Eduardo Miranda. Submitted for possible publication
    in Bulletin of Earthquake Engineering, 2018.

    :param uncertainty_multiplier:
        Value to be multiplied by the uncertainty in the correlation parameter
        beta. If uncertainty_multiplier = 0 (default), the median value is
        used as a constant value.
    """

    def __init__(self, uncertainty_multiplier=0):
        self.uncertainty_multiplier = uncertainty_multiplier
        self.distance_matrix = {}
        self.cache = {}

    def _get_correlation_matrix(self, sites, imt):
        return hmcorrelation(sites, imt, self.uncertainty_multiplier)

    def apply_correlation(self, sites, imt, residuals, stddev_intra):
        """
        Apply correlation to randomly sampled residuals
        """
        # TODO: the case of filtered sites is probably managed incorrectly
        # NB: this is SLOW and we cannot use the cache as in JB2009 because
        # we are not using the complete site collection
        nsites = len(sites)
        assert len(residuals) == len(stddev_intra) == nsites
        D = numpy.diag(stddev_intra)  # phi as a diagonal matrix

        if self.uncertainty_multiplier == 0:   # No uncertainty

            # residuals were sampled from a normal distribution with
            # stddev_intra standard deviation. 'residuals_norm' are residuals
            # normalized, sampled from a standard normal distribution.
            # For this, every row of 'residuals' (every site) is divided by its
            # corresponding standard deviation element.
            residuals_norm = residuals / stddev_intra[:, None]

            # Lower diagonal of the Cholesky decomposition
            # Note that instead of computing the whole correlation matrix
            # corresponding to sites.complete, here we compute only the
            # correlation matrix corresponding to sites
            cormaLow = numpy.linalg.cholesky(
                D @ self._get_correlation_matrix(sites, imt) @ D)

            # Apply correlation
            return cormaLow @ residuals_norm

        else:   # Variability (uncertainty) is included
            nsim = residuals.shape[1]

            # Re-sample all the residuals
            residuals_correlated = residuals * 0
            for isim in range(0, nsim):
                # FIXME: the seed is not set!
                corma = self._get_correlation_matrix(sites, imt)
                # NB: corma is different at each loop since contains
                # randomicity
                residuals_correlated[0:, isim] = (
                    numpy.random.multivariate_normal(
                        numpy.zeros(nsites), D @ corma @ D, 1))

            return residuals_correlated


def hmcorrelation(sites_or_distances, imt, uncertainty_multiplier=0):
    """
    Returns the Heresi-Miranda correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type (PGA or SA)
    :param uncertainty_multiplier:
        Value to be multiplied by the uncertainty in the correlation parameter
        beta. If uncertainty_multiplier = 0 (default), the median value is
        used as a constant value.
    """
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    period = imt.period

    # Eq. (9)
    if period < 1.37:
        Med_b = 4.231 * period * period - 5.180 * period + 13.392
    else:
        Med_b = 0.140 * period * period - 2.249 * period + 17.050

    # Eq. (10)
    Std_b = (4.63e-3 * period*period + 0.028 * period + 0.713)

    # Obtain realization of b
    if uncertainty_multiplier == 0:
        beta = Med_b
    else:
        beta = numpy.random.lognormal(
            numpy.log(Med_b), Std_b * uncertainty_multiplier)

    # Eq. (8)
    res = numpy.exp(-numpy.power((distances / beta), 0.55))
    return res


class EI2012CorrelationModel(BaseCorrelationModel):
    """
    "Spatial Correlation of Spectral Acceleration in European Data"
    by Simon Esposito and Iunio Iervolino. Published in Bulletin of the
    Seismological Society of America, Vol. 102, No. 6, pp. 2781–2788,
    December 2012, doi: 10.1785/0120120068

    :param database:
        Binary input to indicate whether "1" or "2" from which database should
        be applied. ``1`` value means that the values showed are expected to be
        from ESM database, and ``2`` means otherwise.
    """

    def __init__(self, database):
        self.database = database
        self.cache = {}  # imt -> correlation model

    def _get_correlation_matrix(self, sites, imt):
        return ei2012correlation(sites, imt, self.database)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`EI2012CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))


def ei2012correlation(sites_or_distances, imt, database=1):
    """
    Returns the Esposito and Iervolino 2012 correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type (SA)
    :param database:
        if 1 calculates for ESM database
        otherwise calculates for ITACA database
    """
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    period = imt.period

    if not (0.1 <= period <= 2.0):
        raise ValueError(
            f"T = {period} is outside the valid range [0.1, 2.0].")

    if database == 1:  # ESD database
        b = 11.7 + 12.7 * period
        rho = numpy.exp(-(3*distances)/b)
        return rho
    else:  # ITACA database
        b = 8.6 + 11.6 * period
        rho = numpy.exp(-(3*distances)/b)
        return rho


class EI2011CorrelationModel(BaseCorrelationModel):
    """
    Esposito, S., & Iervolino, I. (2011). 
    PGA and PGV Spatial Correlation Models Based on European Multievent Datasets. 
    Bulletin of the Seismological Society of America, 101(5), 2532–2541. 
    https://doi.org/10.1785/0120110117

    :param database:
        Boolean value to indicate whether "1" or "2" from which database should be 
        applied. ``1`` value means that the values showed are expected to be
        from ESM database, and ``2`` means otherwise.
    """
    def __init__(self, database):
        self.database = database
        self.cache = {}  # imt -> correlation model

    def _get_correlation_matrix(self, sites, imt):
        return ei2011correlation(sites, imt, self.database)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`EI2011CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))

def ei2011correlation(sites_or_distances, imt, database=1):
    """
    Returns the Esposito and Iervolino 2011 correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type (PGA or PGV)
    :param database:
        if 1 calculates for ESM database
        otherwise calculates for ITACA database
    """
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    if imt.string == 'PGA':
        if database == 1:  # ESD database
            b = 13.5
            rho = 1 - (1 - numpy.exp(-(3*distances)/b))
            return rho
        else: # ITACA database
            b = 11.5
            rho = 1 - (1 - numpy.exp(-(3*distances)/b))
            return rho
    
    elif imt.string == 'PGV':
        if database == 1:  # ESD database
            b = 21.5
            rho = 1 - (1 - numpy.exp(-(3*distances)/b))
            return rho
        else: # ITACA database
            b = 14.5
            rho = 1 - (1 - numpy.exp(-(3*distances)/b))
            return rho
    else:
        raise ValueError(f"IMT = {imt} is not the appropriate IMT for this model.")

class AHP2022CorrelationModel(BaseCorrelationModel):
    """
    Compute spatial correlation coefficients for Sa(T) and PGA for
    Chilean earthquakes for periods between [0.0-10.0]s.
    
    "Aldea, S., Heresi, P., & Pastén, C. (2022). 
    Within‐event spatial correlation of peak ground acceleration and spectral 
    pseudo‐acceleration ordinates in the Chilean subduction zone. 
    Earthquake Engineering & Structural Dynamics, 51(11), 2575–2590.
    https://doi.org/10.1002/eqe.3674
    
    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type (PGA or SA)
    """
    
    def __init__(self):
        self.cache = {}  # imt -> correlation model

    def _get_correlation_matrix(self, sites, imt):
        return ahp2022correlation(sites, imt)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`AHP2022CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))

def ahp2022correlation(sites_or_distances, imt):
    """
    Returns the Aldea et al., 2022 correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type (PGA or SA)
    """
    
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    period = imt.period

    if not (0 <= period <= 10.0):
        raise ValueError(f"T = {period} is outside the valid range [0, 10.0].")
    
    if period <= 0.40:
        b = 14.400 - 17.00 * period
    elif 0.40 < period <= 0.75:
        b = 14.743 + 7.795 * numpy.log(period)
    elif 0.75 < period <= 3.00:
        b = 12.500
    else:
        b = 5.063 + 6.769 * numpy.log(period)

    rho = numpy.exp(-(distances/b) ** 0.59)

    return rho

class S2022CorrelationModel(BaseCorrelationModel):
    """
    Compute spatial correlation coefficients for Sa(T) and PGA for
    different regions in Italy for periods between [0.0-2.0]s.

    For more details please see: 
    Schiappapietra, E., Stripajová, S., Pažák, P., Douglas, J., & Trendafiloski, G. (2022).
    Exploring the impact of spatial correlations of earthquake ground motions in
    the catastrophe modelling process: a case study for Italy.
    Bulletin of Earthquake Engineering, 20(11), 5747–5773. https://doi.org/10.1007/s10518-022-01413-z
    
    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type (PGA or SA)
    """
    
    def __init__(self, region):
        self.region = region
        self.cache = {}  # imt -> correlation model

    def _get_correlation_matrix(self, sites, imt):
        return s2022correlation(sites, imt, self.region)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`S2022CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))

def s2022correlation(sites_or_distances, imt, region=1):
    """
    Returns the Schiappapietra et al., 2022 correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type (PGA or SA)
    :param region:
        Database used for the spatial correlation:
            - North Italy   - 1
            - Central Italy - 2
            - South Italy   - 3
    """
    
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    period = imt.period

    if not (0 <= period <= 2.0):
        raise ValueError(f"T = {period} is outside the valid range [0, 2.0].")
    
    if region == 1:
        if period <= 0.55:
            b = 27.48 - 52.20 * (period-0.55)
        else:
            b = 27.48 + 15.81 * (period-0.55)
        
        return numpy.exp(-(3*distances)/b)
    
    elif region == 2:
        if period <= 1.0:
            b = 17.87 - 8.52 * (period-1.0)
        else:
            b = 17.87 + 7.85 * (period-1.0)
        
        return numpy.exp(-(3*distances)/b)

    elif region == 3:
        b = 23.25 - 5.44 * period
    
        return numpy.exp(-(3*distances)/b)
    

    
class S2010CorrelationModel(BaseCorrelationModel):
    """
    Compute spatial correlation coefficients for PGA for
    Taiwanese earthquakes.

    For more details please see: 
    Sokolov, V., Wenzel, F., Jean, W.-Y., & Wen, K.-L. (2010). 
    Uncertainty and Spatial Correlation of Earthquake Ground Motion in Taiwan. 
    Terrestrial, Atmospheric and Oceanic Sciences, 21(6), 905. 
    https://doi.org/10.3319/TAO.2010.05.03.01(T)
    
    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type: PGA
    """
    
    def __init__(self):
        self.cache = {}  # imt -> correlation model

    def _get_correlation_matrix(self, sites, imt):
        return s2010correlation(sites, imt)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`S2010CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))

def s2010correlation(sites_or_distances, imt):
    """
    Returns the Sokolov, V. et al., 2010 correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type: PGA
    """
    
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    if imt.string == 'PGA':
        a = -0.586
        b = 0.306
        rho = numpy.exp(a*(distances**b))
        return rho
    else:
        raise ValueError(f"IMT = {imt} is not the appropriate IMT for this model.")
    

class SW2013CorrelationModel(BaseCorrelationModel):
    """
    Compute spatial correlation coefficients for PGA and PGV
    for Japanese earthquakes.

    For more details please see: 
    Sokolov, V., & Wenzel, F. (2013). 
    Further analysis of the influence of site conditions and earthquake 
    magnitude on ground-motion within-earthquake correlation: analysis of 
    PGA and PGV data from the K-NET and the KiK-net (Japan) networks. 
    Bulletin of Earthquake Engineering, 11(6), 1909–1926. https://doi.org/10.1007/s10518-013-9493-9

    
    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type: PGA
    """
    
    def __init__(self):
        self.cache = {}  # imt -> correlation model

    def _get_correlation_matrix(self, sites, imt):
        return sw2013correlation(sites, imt)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`Sw2013CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))

def sw2013correlation(sites_or_distances, imt):
    """
    Returns the Sokolov, V., & Wenzel, F. (2013) correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type: PGA and PGV
    """
    
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances

    if imt.string == 'PGA':
        b = 0.7156
        a = -0.0856
        rho = numpy.exp(a*(distances**b))
        return rho
    elif imt.string == 'PGV':
        b = 0.784
        a = -0.0837
        rho = numpy.exp(a*(distances**b))
        return rho
    else:
        raise ValueError(f"IMT = {imt} is not the appropriate IMT for this model.")
    

    
class DW2013CorrelationModel(BaseCorrelationModel):
    """
    Compute spatial correlation coefficients for Sa(T) and PGA 
    for NGA-W1 database database for periods between [0.2-5.0]s.

    For more details please see: 
    Du, W., & Wang, G. (2013). 
    Intra-Event Spatial Correlations for Cumulative Absolute Velocity, Arias Intensity, 
    and Spectral Accelerations Based on Regional Site Conditions. Bulletin of the Seismological 
    Society of America, 103(2A), 1117–1129. https://doi.org/10.1785/0120120185
    
    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type: SA(T) and PGA
    """
    
    def __init__(self, beta_vs30):
        self.cache = {}  # imt -> correlation model
        self.beta_vs30 = beta_vs30

    def _get_correlation_matrix(self, sites, imt):
        return dw2013correlation(sites, imt, self.beta_vs30)

    def get_lower_triangle_correlation_matrix(self, sites, imt):
        """
        Get lower-triangle matrix as a result of Cholesky-decomposition
        of correlation matrix.

        The resulting matrix should have zeros on values above
        the main diagonal.

        The actual implementations of :class:`BaseCorrelationModel` interface
        might calculate the matrix considering site collection and IMT (like
        :class:`DW013CorrelationModel` does) or might have it pre-constructed
        for a specific site collection and IMT, in which case they will need
        to make sure that parameters to this function match parameters that
        were used to pre-calculate decomposed correlation matrix.

        :param sites:
            :class:`~openquake.hazardlib.site.SiteCollection` to create
            correlation matrix for.
        :param imt:
            Intensity measure type object, see :mod:`openquake.hazardlib.imt`.
        """
        return numpy.linalg.cholesky(self._get_correlation_matrix(sites, imt))

periods = numpy.array([0.2, 0.5, 1.0, 2.0, 5.0])
params = numpy.array([
    [4.4, 1.1],
    [8.5, 1.1],
    [22.8, 0.8],
    [32.3, 0.5],
    [41.4, 0.4],
])

def dw2013correlation(sites_or_distances, imt, beta_vs30):
    """
    Returns the Du, W., & Wang, G. (2013) correlation model.

    :param sites_or_distances:
        SiteCollection instance o distance matrix
    :param imt:
        Intensity Measure Type: PGA
    :param beta_vs30:
        correlation range of the Vs30
    """
    
    if hasattr(sites_or_distances, 'mesh'):
        distances = sites_or_distances.mesh.get_distance_matrix()
    else:
        distances = sites_or_distances
    
    period = imt.period

    # Interpolate the parameters
    interps = [interp1d(periods, params[:, i], kind='linear', fill_value='extrapolate') 
            for i in range(2)]

    if imt.string == 'PGA':
        beta = 7.45 * numpy.exp(0.07 * beta_vs30)
        rho = numpy.exp((-3 * distances) / beta)
        return rho
    
    else:
        if not (0.2 <= period <= 5.0):
            raise ValueError(f"Period = {period} is outside the valid range [0.2, 5.0].")
        
        c1, c2 =[f(period) for f in interps]

        beta = c1 + c2 * beta_vs30
        rho = numpy.exp((-3 * distances) / beta)
        return rho
    
