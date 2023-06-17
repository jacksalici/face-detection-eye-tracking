
import numpy as np
from scipy.spatial.distance import directed_hausdorff as dh




    

def hausdorff_distances (set_a: np.ndarray, set_b: np.ndarray):
    
    n_images = set_a.shape[0]
    assert n_images == set_b.shape[0]
    
    distances = np.empty(n_images)

    for index in range (0, n_images):
        d1, _, _ = dh(set_a[index], set_b[index])
        d2, _, _ = dh(set_b[index], set_a[index])
        distances[index] = max(d1, d2)
        
    return distances
    
def remove_outliers(data, threshold=5):
    mean = np.mean(data)
    std = np.std(data)

    z_scores = (data - mean) / std

    outlier_indices = np.abs(z_scores) > threshold

    cleaned_data = data[~outlier_indices]

    return cleaned_data


points = np.load('test/test_results_backup.npz')["pupils_position"]
# number of samples [1521], method [3], eye [2], coordindates [2]- methods are resp.: ground truth, means of gradients, filtering


n_images = points.shape[0]

set_gt = points[:, 0, :, :]  # (1521, 2, 2)

setp = {}
setp["means_of_gradients"] = points[:, 1, :, :]
setp["filtering"] = points[:, 2, :, :]


for key, value in setp.items():
    distances = hausdorff_distances(set_gt, value)
    cleaned_distances = remove_outliers(distances)
    print(f"{str(key).upper()}\nRemoved outliers: {distances.shape[0]-cleaned_distances.shape[0]}\nMean: {np.mean(cleaned_distances)}\nStd Dev: {np.std(cleaned_distances)}")
    

