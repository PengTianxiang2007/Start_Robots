"""Batch-estimate target-to-camera transforms from ArUco/AprilTag images.

This script loads all PNG/JPG images inside the provided directory, detects a
single AprilTag/ArUco marker (default: DICT_APRILTAG_36h11), and estimates the
pose of the marker relative to the camera. The resulting per-image poses plus
an aggregated average transform are written to disk and echoed to stdout.

Example:

	python camCalibration/world2cam_cs.py \
		--image-dir camCalibration/data_20251111/images \
		--output camCalibration/data_20251111/target2cam_result.json

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import cv2.aruco as aruco
import numpy as np


CAMERA_MODELS = {
	"D435": {
		"camera_matrix": np.array(
			[[610.443, 0.0, 319.213], [0.0, 610.4, 236.05], [0.0, 0.0, 1.0]], dtype=np.float32
		),
		"dist_coeffs": np.array([0, 0, 0, 0, 0], dtype=np.float32),
	},
	"D455": {
		"camera_matrix": np.array(
			[[385.261, 0.0, 322.049], [0.0, 384.677, 244.309], [0.0, 0.0, 1.0]], dtype=np.float32
		),
		"dist_coeffs": np.array(
			[-0.0546432, 0.0645082, -0.000162791, 0.000968451, -0.0208914], dtype=np.float32
		),
	},
}


# --- math helpers ----------------------------------------------------------------


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
	"""Convert a 3x3 rotation matrix to a normalized quaternion (w, x, y, z)."""

	m = np.asarray(R, dtype=float)
	if m.shape != (3, 3):  # pragma: no cover - guard rail
		raise ValueError("Rotation matrix must be 3x3")

	trace = np.trace(m)
	if trace > 0:
		s = np.sqrt(trace + 1.0) * 2
		w = 0.25 * s
		x = (m[2, 1] - m[1, 2]) / s
		y = (m[0, 2] - m[2, 0]) / s
		z = (m[1, 0] - m[0, 1]) / s
	else:
		idx = np.argmax(np.diag(m))
		if idx == 0:
			s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
			w = (m[2, 1] - m[1, 2]) / s
			x = 0.25 * s
			y = (m[0, 1] + m[1, 0]) / s
			z = (m[0, 2] + m[2, 0]) / s
		elif idx == 1:
			s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
			w = (m[0, 2] - m[2, 0]) / s
			x = (m[0, 1] + m[1, 0]) / s
			y = 0.25 * s
			z = (m[1, 2] + m[2, 1]) / s
		else:
			s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
			w = (m[1, 0] - m[0, 1]) / s
			x = (m[0, 2] + m[2, 0]) / s
			y = (m[1, 2] + m[2, 1]) / s
			z = 0.25 * s

	quat = np.array([w, x, y, z], dtype=float)
	quat /= np.linalg.norm(quat)
	return quat


def quaternion_to_rotation_matrix(quat: Sequence[float]) -> np.ndarray:
	"""Convert a quaternion (w, x, y, z) back to a 3x3 rotation matrix."""

	w, x, y, z = quat / np.linalg.norm(quat)
	return np.array(
		[
			[1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
		]
	)


def average_rotations(rotations: Iterable[np.ndarray]) -> np.ndarray:
	"""Average multiple rotation matrices via quaternion-eigen decomposition."""

	quats = np.array([rotation_matrix_to_quaternion(R) for R in rotations])
	accumulator = np.zeros((4, 4), dtype=float)
	for q in quats:
		accumulator += np.outer(q, q)
	eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
	avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
	return quaternion_to_rotation_matrix(avg_quat)


# --- pose estimation -------------------------------------------------------------


@dataclass
class Detection:
	image_name: str
	marker_id: int
	rvec: np.ndarray
	tvec: np.ndarray
	rotation: np.ndarray
	reproj_rmse_px: float
	per_corner_errors_px: np.ndarray

	@property
	def transform(self) -> np.ndarray:
		T = np.eye(4, dtype=float)
		T[:3, :3] = self.rotation
		T[:3, 3] = self.tvec.reshape(3)
		return T


def project_points_manual(
	obj_points: np.ndarray,
	rvec: np.ndarray,
	tvec: np.ndarray,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
) -> np.ndarray:
	"""Project 3D points into the image plane without cv2.projectPoints."""
	obj = np.asarray(obj_points, dtype=float).reshape(-1, 3)
	rvec = np.asarray(rvec, dtype=float).reshape(3)
	tvec = np.asarray(tvec, dtype=float).reshape(3)
	camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
	dist = np.asarray(dist_coeffs, dtype=float).reshape(-1) if dist_coeffs is not None else np.zeros(5)
	if dist.size < 5:
		dist = np.pad(dist, (0, 5 - dist.size))
	k1, k2, p1, p2, k3 = dist[:5]

	rotation, _ = cv2.Rodrigues(rvec)
	cam_pts = (rotation @ obj.T).T + tvec  # (N, 3)
	z = cam_pts[:, 2]
	z = np.where(np.abs(z) < 1e-9, np.sign(z) * 1e-9 + (z == 0) * 1e-9, z)
	x = cam_pts[:, 0] / z
	y = cam_pts[:, 1] / z
	r2 = x**2 + y**2
	radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
	x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x**2)
	y_tan = p1 * (r2 + 2.0 * y**2) + 2.0 * p2 * x * y
	x_dist = x * radial + x_tan
	y_dist = y * radial + y_tan
	fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
	cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
	skew = camera_matrix[0, 1]
	u = fx * x_dist + skew * y_dist + cx
	v = fy * y_dist + cy
	return np.stack([u, v], axis=1)


def compute_reprojection_error(
	obj_points: np.ndarray,
	image_corners: np.ndarray,
	rvec: np.ndarray,
	tvec: np.ndarray,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
	"""Project obj_points into the image and measure pixel errors vs detections."""
	projected = project_points_manual(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
	image_pts = image_corners.reshape(-1, 2)
	deltas = projected - image_pts
	per_point = np.linalg.norm(deltas, axis=1)
	rmse = float(np.sqrt(np.mean(per_point**2)))
	return rmse, per_point, projected


def project_axes_points(
	rvec: np.ndarray,
	tvec: np.ndarray,
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
	axis_length_m: float,
) -> np.ndarray:
	axes = np.array(
		[
			[0.0, 0.0, 0.0],
			[axis_length_m, 0.0, 0.0],
			[0.0, axis_length_m, 0.0],
			[0.0, 0.0, axis_length_m],
		],
		dtype=np.float32,
	)
	img_pts = project_points_manual(axes, rvec, tvec, camera_matrix, dist_coeffs)
	return img_pts.astype(np.float32)

def draw_chessboard_detection(
    image: np.ndarray,
    detected_corners: np.ndarray,
    reprojected_corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    reproj_rmse_px: float,
    axis_length_m: float,
) -> np.ndarray:
    vis = image.copy()

    # Draw detected corners
    cv2.drawChessboardCorners(vis, (detected_corners.shape[0], 1), detected_corners, True)

    # Ensure detected_corners and reprojected_corners have the correct shape
    if len(detected_corners.shape) == 3:
        detected_corners = detected_corners.reshape(-1, 2)
    if len(reprojected_corners.shape) == 3:
        reprojected_corners = reprojected_corners.reshape(-1, 2)

    # Draw reprojection errors
    for d_pt, p_pt in zip(detected_corners, reprojected_corners):
        d_pt_i = tuple(np.round(d_pt).astype(int))
        p_pt_i = tuple(np.round(p_pt).astype(int))

        # Ensure d_pt_i and p_pt_i are tuples of length 2
        if len(d_pt_i) != 2 or len(p_pt_i) != 2:
            print(f"Invalid point coordinates: {d_pt_i}, {p_pt_i}")
            continue

        cv2.circle(vis, d_pt_i, 2, (0, 255, 0), -1)
        cv2.circle(vis, p_pt_i, 2, (255, 0, 0), 2)
        cv2.line(vis, d_pt_i, p_pt_i, (128, 128, 128), 1)

    # Draw coordinate axes
    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)

    text = f"Chessboard t={tvec.reshape(3)} RMSE={reproj_rmse_px:.3f}px"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return vis

# def draw_detection(
# 	image: np.ndarray,
# 	corners: np.ndarray,
# 	camera_matrix: np.ndarray,
# 	dist_coeffs: np.ndarray,
# 	rvec: np.ndarray,
# 	tvec: np.ndarray,
# 	marker_id: int,
# 	reprojected_corners: np.ndarray,
# 	reproj_rmse_px: float,
# 	axis_length_m: float,
# ) -> np.ndarray:
# 	vis = image.copy()
	
# 	# marker_corners = corners.copy()
# 	print(corners)
# 	# shuffle_idx = np.random.permutation(marker_corners.shape[1])
# 	# marker_corners = marker_corners[:, shuffle_idx, :]
# 	aruco.drawDetectedMarkers(vis, [corners])
# 	cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)

# 	detected_pts = corners.reshape(-1, 2)
# 	corner_palette = [
# 		(0, 0, 255),
# 		(0, 180, 0),
# 		(180, 0, 0),
# 		(0, 200, 200),
# 		(200, 0, 200),
# 		(0, 165, 255),
# 	]
# 	for idx, (d_pt, p_pt) in enumerate(zip(detected_pts, reprojected_corners)):
# 		color = corner_palette[idx % len(corner_palette)]
# 		proj_color = tuple(min(255, int(c + 60)) for c in color)
# 		d_pt_i = tuple(np.round(d_pt).astype(int))
# 		p_pt_i = tuple(np.round(p_pt).astype(int))
# 		cv2.circle(vis, d_pt_i, 5, color, -1, cv2.LINE_AA)
# 		cv2.circle(vis, p_pt_i, 5, proj_color, 2, cv2.LINE_AA)
# 		cv2.line(vis, d_pt_i, p_pt_i, (200, 200, 200), 1, cv2.LINE_AA)
# 		label_pos = (d_pt_i[0] + 6, d_pt_i[1] - 6)
# 		cv2.putText(
# 			vis,
# 			str(idx),
# 			label_pos,
# 			cv2.FONT_HERSHEY_SIMPLEX,
# 			0.5,
# 			color,
# 			1,
# 			cv2.LINE_AA,
# 		)

# 	axis_pts = project_axes_points(rvec, tvec, camera_matrix, dist_coeffs, axis_length_m)
# 	origin = tuple(np.round(axis_pts[0]).astype(int))
# 	axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
# 	for idx, color in enumerate(axis_colors, start=1):
# 		endpoint = tuple(np.round(axis_pts[idx]).astype(int))
# 		cv2.line(vis, origin, endpoint, color, 2, cv2.LINE_AA)

# 	text = f"ID:{marker_id} t={tvec.reshape(3)} RMSE={reproj_rmse_px:.3f}px"
# 	cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
# 	return vis


def detect_pose(
    image_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    checkerboard_size: tuple[int, int],  # e.g., (12, 9)
    square_size_m: float,                # e.g., 0.02
    annotate: bool = False,
) -> tuple[Detection, Optional[np.ndarray]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(
        gray,
        checkerboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    )

    if not found:
        raise RuntimeError(f"No chessboard detected in {image_path.name}")

    # Refine corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Define 3D object points (in world frame, Z=0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_m  # scale to meters

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
    if not success:
        raise RuntimeError(f"solvePnP failed for {image_path.name}")

    rotation, _ = cv2.Rodrigues(rvec)

    # Compute reprojection error
    reproj_rmse_px, per_corner_errors_px, reprojected_corners = compute_reprojection_error(
        objp, corners_refined, rvec, tvec, camera_matrix, dist_coeffs
    )

    vis = None
    if annotate:
        vis = draw_chessboard_detection(
            image,
            corners_refined,
            reprojected_corners,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            reproj_rmse_px,
            axis_length_m=square_size_m * 2,
        )

    return (
        Detection(
            image_name=image_path.name,
            marker_id=-1,  # no ID for chessboard
            rvec=rvec.reshape(3),
            tvec=tvec.reshape(3),
            rotation=rotation,
            reproj_rmse_px=reproj_rmse_px,
            per_corner_errors_px=per_corner_errors_px,
        ),
        vis,
    )


def save_results(detections: List[Detection], output_path: Path, meta: dict) -> None:
	aggregate_rotation = average_rotations([d.rotation for d in detections])
	aggregate_translation = np.mean([d.tvec for d in detections], axis=0)

	aggregate_transform = np.eye(4, dtype=float)
	aggregate_transform[:3, :3] = aggregate_rotation
	aggregate_transform[:3, 3] = aggregate_translation

	payload = {
		**meta,
		"detections": [
			{
				"image": d.image_name,
				"marker_id": d.marker_id,
				"rvec": d.rvec.tolist(),
				"tvec_m": d.tvec.tolist(),
				"rotation_matrix": d.rotation.tolist(),
				"reproj_rmse_px": d.reproj_rmse_px,
				"per_corner_errors_px": d.per_corner_errors_px.tolist(),
				"transform_matrix": d.transform.tolist(),
			}
			for d in detections
		],
		"aggregate": {
			"rotation_matrix": aggregate_rotation.tolist(),
			"translation_m": aggregate_translation.tolist(),
			"transform_matrix": aggregate_transform.tolist(),
		},
	}

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as fp:
		json.dump(payload, fp, indent=2)

	print("Saved results to", output_path)
	print("Aggregate target->cam transform (row-major):")
	np.set_printoptions(precision=6, suppress=True)
	print(aggregate_transform)


# --- CLI ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--image-dir",
		type=Path,
		default=Path("camCalibration/data_20251111/images"),
		help="Directory containing input images (.png/.jpg).",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("camCalibration/data_20251111/target2cam_result.json"),
		help="Where to store the JSON summary.",
	)
	parser.add_argument(
		"--checkerboard-cols",
		type=int,
		default=11,
		help="Number of inner corners along width (e.g., 12 for 12x9 board).",
	)
	parser.add_argument(
		"--checkerboard-rows",
		type=int,
		default=8,
		help="Number of inner corners along height.",
	)
	parser.add_argument(
		"--square-size",
		type=float,
		default=0.02,
		help="Size of one square in meters (e.g., 0.02 for 20mm).",
	)

	# parser.add_argument(
	# 	"--marker-size",
	# 	type=float,
	# 	default=0.076,
	# 	help="Physical marker size (edge length in meters).",
	# )
	# parser.add_argument(
	# 	"--dictionary",
	# 	type=str,
	# 	default="APRILTAG_36h11",
	# 	help="OpenCV predefined dictionary name.",
	# )
	parser.add_argument(
		"--intrinsic",
		type=Path,
		default=None,
		help="Optional path to a 3x3 intrinsic matrix stored as .npy or .json.",
	)
	parser.add_argument(
		"--distortion",
		type=Path,
		default=None,
		help="Optional path to distortion coefficients stored as .npy or .json.",
	)
	parser.add_argument(
		"--camera-model",
		choices=sorted(CAMERA_MODELS.keys()),
		default="D455",
		help="Preset intrinsics to use when no intrinsic/distortion files are provided.",
	)
	parser.add_argument(
		"--save-vis-dir",
		type=Path,
		default=None,
		help="Directory to dump annotated detection images (PNG).",
	)
	parser.add_argument(
		"--show-vis",
		action="store_true",
		help="Display each annotated detection in a window.",
	)
	parser.add_argument(
		"--show-wait-ms",
		type=int,
		default=800,
		help="Delay in milliseconds for each displayed visualization window.",
	)
	return parser.parse_args()


def load_matrix(path: Path, expected_shape: Sequence[int]) -> np.ndarray:
	if path.suffix.lower() == ".npy":
		data = np.load(path)
	else:
		with path.open("r", encoding="utf-8") as fp:
			data = np.array(json.load(fp))
	arr = np.asarray(data, dtype=float)
	if tuple(arr.shape) != tuple(expected_shape):  # pragma: no cover
		raise ValueError(f"Expected shape {expected_shape}, got {arr.shape}")
	return arr


def load_vector(path: Path) -> np.ndarray:
	if path.suffix.lower() == ".npy":
		data = np.load(path)
	else:
		with path.open("r", encoding="utf-8") as fp:
			data = np.array(json.load(fp))
	arr = np.asarray(data, dtype=float).reshape(-1)
	return arr


def main() -> None:
	args = parse_args()

	if not args.image_dir.exists():
		raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

	image_paths = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
	if not image_paths:
		raise FileNotFoundError(f"No images found in {args.image_dir}")

	defaults = CAMERA_MODELS[args.camera_model]
	camera_matrix = defaults["camera_matrix"].copy()
	dist_coeffs = defaults["dist_coeffs"].copy()

	if args.intrinsic:
		camera_matrix = load_matrix(args.intrinsic, (3, 3))
	if args.distortion:
		dist_coeffs = load_vector(args.distortion)

	need_visuals = bool(args.save_vis_dir or args.show_vis)
	if args.save_vis_dir:
		args.save_vis_dir.mkdir(parents=True, exist_ok=True)

	# dict_name = args.dictionary
	# if not dict_name.startswith("DICT_"):
	# 	dict_name = f"DICT_{dict_name}"
	# if not hasattr(aruco, dict_name):
	# 	raise ValueError(f"Unknown ArUco/AprilTag dictionary: {args.dictionary}")
	# dict_id = getattr(aruco, dict_name)
	# dictionary = aruco.getPredefinedDictionary(dict_id)
	# detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

	# checkerboard_size = (12, 9)   # <-- YOUR VALUE HERE
	# square_size_m = 0.02          # 20 mm

	checkerboard_size = (args.checkerboard_cols, args.checkerboard_rows)
	square_size_m = args.square_size

	detections: List[Detection] = []
	for path in image_paths:
		try:
			detection, vis = detect_pose(
				path,
				camera_matrix,
				dist_coeffs,
				checkerboard_size,
				square_size_m,
				annotate=need_visuals,
			)
		except Exception as exc:
			print(f"[WARN] Failed on {path.name}: {exc}")
			continue
		detections.append(detection)
		print(
			f"{path.name} -> t = {detection.tvec}, rvec = {detection.rvec}, RMSE = {detection.reproj_rmse_px:.3f}px"
		)
		if vis is not None:
			if args.save_vis_dir:
				vis_path = args.save_vis_dir / f"{path.stem}_vis.png"
				cv2.imwrite(str(vis_path), vis)
				print(f"  saved visualization -> {vis_path}")
			if args.show_vis:
				cv2.imshow("target2cam", vis)
				cv2.waitKey(max(args.show_wait_ms, 1))

	if not detections:
		raise RuntimeError("No valid detections were produced.")

	if args.show_vis:
		cv2.destroyAllWindows()

	# meta = {
	# 	"image_dir": str(args.image_dir),
	# 	"marker_size_m": args.marker_size,
	# 	"dictionary": dict_name,
	# 	"camera_matrix": camera_matrix.tolist(),
	# 	"dist_coeffs": dist_coeffs.tolist(),
	# }

	meta = {
		"image_dir": str(args.image_dir),
		# 如果需要，可以添加棋盘格单个方块的边长，单位为米
		"square_size_m": args.square_size,  # 假设你通过命令行参数提供了这个值
		# 可以移除或注释掉下面两行，如果它们对你的应用不适用的话
		# "marker_size_m": args.marker_size,
		# "dictionary": dict_name,
		"camera_matrix": camera_matrix.tolist(),
		"dist_coeffs": dist_coeffs.tolist(),
	}
	save_results(detections, args.output, meta)

if __name__ == "__main__":
	main()
