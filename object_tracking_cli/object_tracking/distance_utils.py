from scipy.spatial import KDTree


def find_unique_closest_pairs(centroids, points):
    centroid_tree = KDTree(centroids)
    closest_centroids_indices = centroid_tree.query(points)[1]
    used_centroid_idxs = set()
    closest_pairs = {}
    for point_idx, centroid_idx in enumerate(closest_centroids_indices):
        # resolve potential conflict
        if centroid_idx in used_centroid_idxs:
            k = 2
            while k < len(centroids):
                next_centroid_idx = centroid_tree.query(points[point_idx], k=k)[1][
                    k - 1
                ]
                if next_centroid_idx not in used_centroid_idxs:
                    closest_pairs[point_idx] = next_centroid_idx
                    used_centroid_idxs.add(next_centroid_idx)
                    break
                k += 1
        else:
            closest_pairs[point_idx] = centroid_idx
            used_centroid_idxs.add(centroid_idx)
    return closest_pairs
