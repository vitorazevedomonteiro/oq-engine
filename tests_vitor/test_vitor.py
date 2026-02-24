import numpy as np
from openquake.hazardlib.correlation.cross_spatial_correlation import LothBaker2013CorrelationModel, MarkhvidaEtAl2018CorrelationModel
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Point
from openquake.hazardlib.imt import SA

SITECOL = SiteCollection([Site(Point(2, -40), 1, 1, 1),
                            Site(Point(2, -40.1), 1, 1, 1),
                            Site(Point(2, -39.9), 1, 1, 1)])

cormo =  MarkhvidaEtAl2018CorrelationModel()
imts = [SA(period=0.5, damping=5), SA(period=3.0, damping=5)]
cormo.get_lower_triangle_correlation_matrix(SITECOL, imts)

lt = cormo.cache["corma"]

print("Shape:", lt.shape)
print(lt)

"""
import numpy as np
from openquake.hazardlib.correlation.cross_spatial_correlation import LothBaker2013CorrelationModel, MarkhvidaEtAl2018CorrelationModel
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Point
from openquake.hazardlib.imt import SA

# Define sites
SITECOL = SiteCollection([
    Site(Point(2, -40), 1, 1, 1),
    Site(Point(2, -40.1), 1, 1, 1),
    Site(Point(2, -39.9), 1, 1, 1)
])

np.random.seed(13)

# Instantiate model
cormo = LothBaker2013CorrelationModel()

# Define IMTs
imts = [SA(period=0.5, damping=5), SA(period=5.0, damping=5)] 

# Filter sites if needed
filtered = SITECOL.filtered([0, 2])
filtered.indices = np.array([0, 2])  # must set for apply_correlation

num_sites_filtered = len(filtered)
num_imts = len(imts)
num_realizations = 5  # small number for inspection

# Sample uncorrelated residuals
uncorr_residuals = np.random.normal(size=(num_sites_filtered * num_imts, num_realizations))

# Apply correlation
intra_residuals_correlated = cormo.apply_correlation(filtered, imts, uncorr_residuals)

# Print results in a way you can copy into aaaa()
print("Shape:", intra_residuals_correlated.shape)
print("Values:")
for imt_idx in range(intra_residuals_correlated.shape[0]):
    print(list(intra_residuals_correlated[imt_idx]))
"""



"""

import numpy
from openquake.hazardlib.correlation.spatial_correlation import EI2012CorrelationModel
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Point
from openquake.hazardlib.imt import SA

SITECOL = SiteCollection([
    Site(Point(2, -40), 1, 1, 1),
    Site(Point(2, -40.1), 1, 1, 1),
    Site(Point(2, -39.9), 1, 1, 1)
])

filtered = SITECOL.filtered([0, 2])

numpy.random.seed(13)
cormo = EI2012CorrelationModel(database=1)
intra_residuals_sampled = numpy.random.normal(size=(2, 5))  # 2 sites × 5 samples

intra_residuals_correlated = cormo.apply_correlation(
    filtered, SA(0.5), intra_residuals_sampled
)

print("Correlated residuals (2x5):")
print(intra_residuals_correlated)
"""