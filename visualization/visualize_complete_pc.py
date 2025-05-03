import os
import open3d as o3d

def load_and_combine_ply(ply_folder):
    pcd_combined = o3d.geometry.PointCloud()
    ply_files = sorted([f for f in os.listdir(ply_folder) if f.endswith(".ply")])

    if not ply_files:
        raise ValueError(f"No .ply files found in {ply_folder}")

    print(f"[INFO] Found {len(ply_files)} .ply files.")

    for i, fname in enumerate(ply_files):
        path = os.path.join(ply_folder, fname)
        print(f"[INFO] Loading: {path}")
        pcd = o3d.io.read_point_cloud(path)
        pcd_combined += pcd

    return pcd_combined

def visualize_combined_pointcloud(pcd):
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    print(f"[INFO] Visualizing {len(pcd.points)} points...")
    o3d.visualization.draw_geometries(
        [pcd, coord],
        window_name="Combined Point Cloud",
        width=1280,
        height=720,
        zoom=0.6
    )

if __name__ == "__main__":
    # 👇 Replace this with your actual directory containing .ply files
    ply_folder = "/home/clearlab/data_plot/data_output/00008/0"

    # Load and combine
    combined_pcd = load_and_combine_ply(ply_folder)

    # Visualize
    visualize_combined_pointcloud(combined_pcd)
