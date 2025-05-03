import open3d as o3d
import numpy as np

# 1. Load & clean
pcd = o3d.io.read_point_cloud("./processed_unreal/00008/1/00000.ply")
pcd = pcd.voxel_down_sample(0.1)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 2. RANSAC plane
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=1000
)
ground = pcd.select_by_index(inliers)
nonground = pcd.select_by_index(inliers, invert=True)

# compute mean ground height
ground_pts = np.asarray(ground.points)
ground_z_mean = float(ground_pts[:, 2].mean())
print(f"[INFO] shifted by {-ground_z_mean:.3f} m → ground at Z=0")

# 3. SHIFT in numpy, then reassign back
#   ground
ground_pts[:, 2] -= ground_z_mean
ground.points = o3d.utility.Vector3dVector(ground_pts)



#   nonground_final (after your height refine)
pts_ng = np.asarray(nonground.points)
mask_low = pts_ng[:, 2] < (ground_z_mean + 0.05)
nonground_final = nonground.select_by_index(
    np.where(~mask_low)[0]  # keep only pts above that threshold
)
nf_pts = np.asarray(nonground_final.points)
nf_pts[:, 2] -= ground_z_mean
nonground_final.points = o3d.utility.Vector3dVector(nf_pts)

# 4. Split by z<0 and color
pts = np.asarray(nonground_final.points)
neg_idx = np.where(pts[:, 2] < 0)[0]
pos_idx = np.where(pts[:, 2] >= 0)[0]

below = nonground_final.select_by_index(neg_idx)
above = nonground_final.select_by_index(pos_idx)

ground.paint_uniform_color([0.1, 0.9, 0.1])    # green
above.paint_uniform_color([0.9, 0.1, 0.1])     # red
below.paint_uniform_color([0.1, 0.1, 0.9])     # blue

# add coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)

# visualize
o3d.visualization.draw_geometries([ground, above, below, frame])
