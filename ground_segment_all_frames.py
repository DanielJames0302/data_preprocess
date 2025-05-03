import numpy as np
import os
import open3d as o3d


def load_frames(data_dir: str, num_frames: int = 100):
    """Read *.ply files, flipping Z."""
    frames = []
    for i in range(num_frames):
        fp = os.path.join(data_dir, f"{i:05d}.ply")
        if not os.path.exists(fp):
            print(f"[WARN] missing file: {fp}")
            continue

        pts = []
        with open(fp, "r") as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == "end_header":
                        in_header = False
                    continue
                x, y, z = map(float, line.split()[:3])
                pts.append([x, y, -z])  # flip Z
        if pts:
            frames.append(np.asarray(pts, dtype=np.float32))
    if not frames:
        raise RuntimeError("No .ply files found – check data_dir!")
    return frames


def save_segmented_frame(
    pts: np.ndarray,
    idx: int,
    output_dir: str,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    frame_size: float = 0.5
):
    """
    Segment ground vs non-ground via RANSAC, shift so ground at z=0,
    color green/red, and save a screenshot for this frame.
    """
    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # RANSAC plane fitting
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    if not inliers:
        print(f"[WARN] Frame {idx:03d}: no ground plane found")
        return

    ground = pcd.select_by_index(inliers)
    nonground = pcd.select_by_index(inliers, invert=True)

    # Compute and apply shift so ground plane at z=0
    ground_z = float(np.mean(np.asarray(ground.points)[:, 2]))
    ground.translate((0, 0, -ground_z))
    nonground.translate((0, 0, -ground_z))

    # Colorize
    ground.paint_uniform_color([0.1, 0.9, 0.1])
    nonground.paint_uniform_color([0.9, 0.1, 0.1])

    # Setup offscreen visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(ground)
    vis.add_geometry(nonground)
    # add a frame at origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    vis.add_geometry(frame)
    vis.poll_events()
    vis.update_renderer()

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"segmented_{idx:05d}.png")
    vis.capture_screen_image(save_path, do_render=True)
    print(f"[✓] Saved segmentation: {save_path}")
    vis.destroy_window()


if __name__ == "__main__":
    data_dir   = "./processed_unreal/00008/1"
    NUM_FRAMES = 199
    out_dir    = os.path.join(data_dir, "segmented_frames")

    frames = load_frames(data_dir, NUM_FRAMES)
    for idx, pts in enumerate(frames):
        save_segmented_frame(
            pts, idx, out_dir,
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000,
            frame_size=0.5
        )
