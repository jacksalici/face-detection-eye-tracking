
import numpy as np
from scipy.spatial.distance import directed_hausdorff

points = np.load('test/test_results_backup.npz')["pupils_position"]

set1 = points[:, 0, :, :].reshape(-1, 2)
set2 = points[:, 1, :, :].reshape(-1, 2)

dist_1_to_2 = directed_hausdorff(set1, set2)[0]
dist_2_to_1 = directed_hausdorff(set2, set1)[0]

hausdorff_distance = max(dist_1_to_2, dist_2_to_1)

print("Hausdorff Distance:", hausdorff_distance)

set1 = points[:, 0, :, :].reshape(-1, 2)
set3 = points[:, 2, :, :].reshape(-1, 2)

dist_1_to_2 = directed_hausdorff(set1, set3)[0]
dist_2_to_1 = directed_hausdorff(set3, set1)[0]

hausdorff_distance = max(dist_1_to_2, dist_2_to_1)

print("Hausdorff Distance:", hausdorff_distance)