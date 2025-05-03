import numpy as np
from PIL import Image
import open3d as o3d

# === FILE PATHS ===
base_dir   = "/home/clearlab/data_plot/processed_unreal/00008/1"
frame_id   = "00000"
rgb_path   = f"{base_dir}/{frame_id}_rgb.png"
depth_path = f"{base_dir}/{frame_id}_depth.npy"
cam_path   = f"{base_dir}/{frame_id}.npz"

# === LOAD DATA ===
rgb      = np.array(Image.open(rgb_path)).astype(np.uint8)
depth    = np.load(depth_path)
cam_data = np.load(cam_path)
K        = cam_data["intrinsics"]

print(f"[INFO] Intrinsics:\n{K}")
print(f"[INFO] RGB shape: {rgb.shape}, Depth shape: {depth.shape}")

# === DEPTH → CAMERA SPACE ===
H, W      = depth.shape
fx, fy    = K[0,0], K[1,1]
cx, cy    = K[0,2], K[1,2]
i, j      = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
z         = depth
x         = (i - cx) * z / fx
y         = (j - cy) * z / fy

points    = np.stack((x, y, z), axis=2).reshape(-1, 3)
valid     = (z.reshape(-1) > 0)
points    = points[valid]
colors    = rgb.reshape(-1, 3)[valid] / 255.0

# === CIRCULAR CROP PARAMETERS ===
RADIUS   = 15.0               # meters
PLANE    = "yz"               # pick "xy", "xz", or "yz"
axis_map = {"xy": (0,1), "xz": (0,2), "yz": (1,2)}
a, b     = axis_map[PLANE]
coords   = points[:, [a, b]]
mask_c   = np.linalg.norm(coords, axis=1) <= RADIUS

points_c = points[mask_c]
colors_c = colors[mask_c]

print(f"[INFO] Cropped to {len(points_c)} points within {RADIUS}m in {PLANE.upper()}-plane")

# === BUILD POINT CLOUD ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_c)
pcd.colors = o3d.utility.Vector3dVector(colors_c)

# === GROUND PLANE SEGMENTATION ===
# RANSAC parameters
dist_th    = 0.02
ransac_n   = 3
n_iters    = 1000
refine_th  = 0.05

plane_model, inliers = pcd.segment_plane(dist_th, ransac_n, n_iters)
if not inliers:
    raise RuntimeError("RANSAC failed to find a ground plane.")

ground_raw    = pcd.select_by_index(inliers)
nonground_raw = pcd.select_by_index(inliers, invert=True)

# Compute mean height of the raw ground plane
ground_z_mean = np.mean(np.asarray(ground_raw.points)[:, 2])

# Pull in any points within refine_th of the plane as ground
pts_ng   = np.asarray(nonground_raw.points)
mask_low = (pts_ng[:, 2] < (ground_z_mean + refine_th))

ground_refined  = nonground_raw.select_by_index(np.where(mask_low)[0])
nonground_final = nonground_raw.select_by_index(np.where(mask_low)[0], invert=True)

# Translate everything so ground plane → z=0
translation = (0.0, 0.0, -ground_z_mean)
for cloud in (ground_raw, ground_refined, nonground_final):
    cloud.translate(translation)

# Merge ground segments & preserve original RGB
ground_pts   = np.vstack((np.asarray(ground_raw.points),
                          np.asarray(ground_refined.points)))
ground_cols  = np.vstack((np.asarray(ground_raw.colors),
                          np.asarray(ground_refined.colors)))
ground_final = o3d.geometry.PointCloud()
ground_final.points = o3d.utility.Vector3dVector(ground_pts)
ground_final.colors = o3d.utility.Vector3dVector(ground_cols)

# === VISUALIZE ===
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=(0,0,0))
o3d.visualization.draw_geometries(
    [ground_final, nonground_final, frame],
    window_name=f"Crop+Segmentation ({PLANE.upper()} ≤ {RADIUS}m)",
    width=1280, height=720, zoom=0.6
)
