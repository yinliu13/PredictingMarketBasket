from scipy.stats import wasserstein_distance
import ot

fml = wasserstein_distance([0, 1, 3], [5, 6, 8])
print(fml)
#ot.wasserstein_1d([0, 1, 3], [5, 6, 8])
