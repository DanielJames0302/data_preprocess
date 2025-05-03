import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def load_and_downsample(ply_path: str, voxel_size: float) -> o3d.geometry.PointCloud:
    """Load a point cloud and apply voxel down-sampling."""
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"No points loaded from {ply_path}")
    return pcd.voxel_down_sample(voxel_size)

def normalize_ground(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000
) -> o3d.geometry.PointCloud:
    """
    1) Run RANSAC to find the dominant planar patch (ground).
    2) Compute its mean Z and shift the cloud so that plane sits at Z=0.
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    if len(inliers) == 0:
        raise RuntimeError("Unable to find any planar inliers for ground.")

    pts = np.asarray(pcd.points)
    ground_z = pts[inliers, 2].mean()

    pcd.translate((0, 0, -ground_z))
    print(f"[INFO] Shifted cloud by {-ground_z:.3f} m so ground at Z=0")
    return pcd

def build_obstacle_map(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    height_threshold: float
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Generate a 2D occupancy map (X–Y) marking voxels
    where Z > height_threshold as obstacles.
    """
    pts = np.asarray(pcd.points)
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    nx = int(np.ceil((x_max - x_min) / voxel_size)) + 1
    ny = int(np.ceil((y_max - y_min) / voxel_size)) + 1

    # Compute grid indices for all points above threshold
    mask = zs > height_threshold
    ix = np.floor((xs[mask] - x_min) / voxel_size).astype(int)
    iy = np.floor((ys[mask] - y_min) / voxel_size).astype(int)

    occ_map = np.zeros((nx, ny), dtype=np.uint8)
    occ_map[ix, iy] = 1  # mark occupied

    return occ_map, x_min, x_max, y_min, y_max

def save_and_plot_map(
    occ_map: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    height_threshold: float,
    output_path: str
):
    """Save and display the 2D obstacle map."""
    plt.figure(figsize=(6,6))
    plt.imshow(
        occ_map.T,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        interpolation='nearest'
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Obstacle Map @ z > {height_threshold} m")
    plt.colorbar(label="Occupied (1) vs Free (0)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved 2D obstacle map to: {output_path}")
    plt.show()

def visualize_3d(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Voxelize the point cloud and visualize it alongside the original data."""
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5,
        origin=(0.0, 0.0, 0.0)
    )
    o3d.visualization.draw_geometries(
        [pcd, vg, frame],
        window_name="Point Cloud + Voxel Grid",
        width=800, height=600
    )

def main():
    ply_path = "./processed_unreal/00008/1/00198.ply"
    voxel_size = 0.1
    height_threshold = 2.0
    output_path = "top_down_obstacle_map_xy.png"

    # 1) Load & downsample
    pcd = load_and_downsample(ply_path, voxel_size)

    # 2) Normalize ground via RANSAC
    pcd = normalize_ground(pcd,
                           distance_threshold=0.02,
                           ransac_n=3,
                           num_iterations=1000)

    # 3) Build 2D obstacle map
    occ_map, x_min, x_max, y_min, y_max = build_obstacle_map(
        pcd, voxel_size, height_threshold
    )

    # 4) Save & plot
    save_and_plot_map(
        occ_map, x_min, x_max, y_min, y_max,
        height_threshold, output_path
    )

    # 5) Visualize in 3D
    visualize_3d(pcd, voxel_size)

if __name__ == "__main__":
    main()
