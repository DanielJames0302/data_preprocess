import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def load_and_downsample(ply_path: str, voxel_size: float) -> o3d.geometry.PointCloud:
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
    1) Run RANSAC to find the largest planar patch (ground).
    2) Compute its average Z and translate the cloud so that
       ground plane lies at Z = 0.
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    if len(inliers) == 0:
        raise RuntimeError("Unable to find any planar inliers for ground.")

    # Points belonging to the plane
    pts = np.asarray(pcd.points)
    ground_z = pts[inliers, 2].mean()

    # Shift so that mean ground height == 0
    pcd.translate((0, 0, -ground_z))
    print(f"[INFO] Shifted cloud by {-ground_z:.3f} m so ground plane at Z = 0")

    return pcd

def build_visibility_map_from_voxel(
    vg: o3d.geometry.VoxelGrid,
    ground_level: int = 0
) -> tuple[np.ndarray, float, float, float, float]:
    # Move CUDA grid back to CPU if needed
    if hasattr(vg, "cpu"):
        vg = vg.cpu()

    # 1) Retrieve the list of voxels
    voxels = vg.get_voxels()
    if not voxels:
        raise RuntimeError("VoxelGrid has no voxels!")

    # 2) Extract their grid indices
    grid_idxs = np.array([v.grid_index for v in voxels], dtype=int)
    ixs, iys, izs = grid_idxs.T
    nx, ny = ixs.max() + 1, iys.max() + 1

    # 3) Mark (i,j) columns with any z > ground_level
    vis_map = np.zeros((nx, ny), dtype=bool)
    above = grid_idxs[izs > ground_level][:, :2]
    if above.size:
        unique_xy = np.unique(above, axis=0)
        vis_map[unique_xy[:, 0], unique_xy[:, 1]] = True

    # 4) Recover world extents
    vs = vg.voxel_size
    x_min, y_min = vg.origin[0], vg.origin[1]
    x_max = x_min + nx * vs
    y_max = y_min + ny * vs

    return vis_map, x_min, x_max, y_min, y_max

def save_and_plot_map(
    map_arr: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    title: str,
    output_path: str
):
    plt.figure(figsize=(6, 6))
    plt.imshow(
        map_arr.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        cmap="gray",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.colorbar(label="1 = visible / 0 = not visible")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved map: {output_path}")
    plt.show()

def main():
    ply_path = "./processed_unreal/00008/1/00198.ply"
    voxel_size = 0.4

    # 1) Load, downsample, then normalize via ground‐plane RANSAC
    pcd = load_and_downsample(ply_path, voxel_size)
    pcd = normalize_ground(pcd,
                           distance_threshold=0.02,
                           ransac_n=3,
                           num_iterations=1000)

    # 2) Build voxel grid
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    # 3) Build visibility map: any voxel at z-index > 0
    vis_map, x_min, x_max, y_min, y_max = build_visibility_map_from_voxel(
        vg, ground_level=0
    )

    # 4) Save & plot
    save_and_plot_map(
        vis_map.astype(np.uint8),
        x_min, x_max, y_min, y_max,
        title="Visibility Map (any voxel above ground)",
        output_path="visibility_map_xy.png"
    )

    # 5) Visualize in 3D with the new ground‐aligned frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    o3d.visualization.draw_geometries([pcd, vg, frame])

if __name__ == "__main__":
    main()
