import numpy as np
import pandas as pd
from scipy.stats import entropy

from DataProcessing.utils import plot_2d_csv


def jensen_shannon_divergence(P, Q):
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    """
    P = np.asarray(P)
    Q = np.asarray(Q)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    M = 0.5 * (P + Q)
    M = M + 1e-10
    return 0.5 * (entropy(P, M) + entropy(Q, M))


def point_to_grid(point, grid_size=16):
    """
    Map a point (x,y) in [0,1]x[0,1] to a grid index in a grid_size x grid_size grid.
    """
    x, y = point
    i = min(grid_size - 1, int(x * grid_size))
    j = min(grid_size - 1, int(y * grid_size))
    return i * grid_size + j


def get_density_counts(trajectories, grid_size=16):
    """
    Compute the histogram of all points across all trajectories on a grid_size x grid_size grid.
    """
    all_points = [point_to_grid(p, grid_size) for traj in trajectories for p in traj]
    hist = np.bincount(all_points, minlength=grid_size ** 2)
    return hist


def get_start_counts(trajectories, grid_size=16):
    """
    Compute the histogram of starting points of all trajectories on a grid_size x grid_size grid.
    """
    start_points = [point_to_grid(traj[0], grid_size) for traj in trajectories if len(traj) > 0]
    hist = np.bincount(start_points, minlength=grid_size ** 2)
    return hist


def get_end_counts(trajectories, grid_size=16):
    """
    Compute the histogram of ending points of all trajectories on a grid_size x grid_size grid.
    """
    end_points = [point_to_grid(traj[-1], grid_size) for traj in trajectories if len(traj) > 0]
    hist = np.bincount(end_points, minlength=grid_size ** 2)
    return hist


def total_length(traj):
    """
    Compute the total Euclidean length of a trajectory.
    """
    return sum(
        np.sqrt((traj[i + 1][0] - traj[i][0]) ** 2 + (traj[i + 1][1] - traj[i][1]) ** 2) for i in range(len(traj) - 1))


def get_length_histogram(trajectories, num_bins=20):
    """
    Compute the histogram of total lengths of all trajectories with num_bins bins.
    """
    lengths = [total_length(traj) for traj in trajectories]
    if not lengths:
        return np.zeros(num_bins)
    min_len = min(lengths)
    max_len = max(lengths)
    bins = np.linspace(min_len, max_len, num_bins + 1)
    hist, _ = np.histogram(lengths, bins=bins)
    return hist


def evaluate_difftraj(real_trajectories, generated_trajectories, grid_size=16, length_bins=20):
    """
    Compute all evaluation metrics for DiffTraj model.

    Parameters:
    - real_trajectories: List of real trajectories, each as a list of (x,y) points.
    - generated_trajectories: List of generated trajectories, each as a list of (x,y) points.
    - grid_size: Size of the grid (default=16).
    - length_bins: Number of bins for length histogram (default=20).

    Returns:
    - A dictionary with metrics: density_error, trip_error, length_error, pattern_score.
    """

    # Density Error
    hist_real_density = get_density_counts(real_trajectories, grid_size)
    hist_gen_density = get_density_counts(generated_trajectories, grid_size)
    P_density_real = hist_real_density / np.sum(hist_real_density)
    P_density_gen = hist_gen_density / np.sum(hist_gen_density)
    density_error = jensen_shannon_divergence(P_density_real, P_density_gen)

    # Trip Error: Average of JSD for start and end points
    hist_real_start = get_start_counts(real_trajectories, grid_size)
    hist_gen_start = get_start_counts(generated_trajectories, grid_size)
    P_start_real = hist_real_start / np.sum(hist_real_start)
    P_start_gen = hist_gen_start / np.sum(hist_gen_start)
    jsd_start = jensen_shannon_divergence(P_start_real, P_start_gen)

    hist_real_end = get_end_counts(real_trajectories, grid_size)
    hist_gen_end = get_end_counts(generated_trajectories, grid_size)
    P_end_real = hist_real_end / np.sum(hist_real_end)
    P_end_gen = hist_gen_end / np.sum(hist_gen_end)
    jsd_end = jensen_shannon_divergence(P_end_real, P_end_gen)
    trip_error = (jsd_start + jsd_end) / 2

    # Length Error
    real_lengths = [total_length(traj) for traj in real_trajectories]
    gen_lengths = [total_length(traj) for traj in generated_trajectories]
    all_lengths = real_lengths + gen_lengths
    if all_lengths:
        min_len = min(all_lengths)
        max_len = max(all_lengths)
        bins = np.linspace(min_len, max_len, length_bins + 1)
    else:
        bins = np.array([0, 1])
    hist_real_length, _ = np.histogram(real_lengths, bins=bins)
    hist_gen_length, _ = np.histogram(gen_lengths, bins=bins)
    P_length_real = hist_real_length / np.sum(hist_real_length) if np.sum(hist_real_length) > 0 else np.zeros_like(
        hist_real_length)
    P_length_gen = hist_gen_length / np.sum(hist_gen_length) if np.sum(hist_gen_length) > 0 else np.zeros_like(
        hist_gen_length)
    length_error = jensen_shannon_divergence(P_length_real, P_length_gen)

    # Pattern Score
    n = 10
    total_hist_real = get_density_counts(real_trajectories, grid_size)
    total_hist_gen = get_density_counts(generated_trajectories, grid_size)
    top_n_real = np.argsort(total_hist_real)[::-1][:n]
    top_n_gen = np.argsort(total_hist_gen)[::-1][:n]
    P_set = set(top_n_real)
    Pgen_set = set(top_n_gen)
    intersection = len(P_set & Pgen_set)
    precision = intersection / len(Pgen_set) if Pgen_set else 0
    recall = intersection / len(P_set) if P_set else 0
    if precision + recall > 0:
        pattern_score = 2 * precision * recall / (precision + recall)
    else:
        pattern_score = 0

    return {
        "density_error": density_error,
        "trip_error": trip_error,
        "length_error": length_error,
        "pattern_score": pattern_score
    }


def load_and_preprocess_trajectories(csv_file):
    """
    Load trajectories from a CSV file and preprocess them for DiffTraj evaluation.

    Parameters:
    - csv_file: Path to the CSV file with columns 'id', 'lon', 'lat'.

    Returns:
    - trajectories: List of trajectories, each a list of (x,y) points normalized to [0,1].
    """
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Ensure required columns exist
    if not {'id', 'lon', 'lat'}.issubset(df.columns):
        raise ValueError("CSV must contain 'id', 'lon', 'lat' columns")

    # Group by 'id' to form trajectories
    trajectories = []
    for traj_id, group in df.groupby('id'):
        # Extract (lon, lat) points, ensuring valid coordinates
        points = group[['lon', 'lat']].dropna().values.tolist()
        if points:  # Only include non-empty trajectories
            trajectories.append(points)

    if not trajectories:
        raise ValueError("No valid trajectories found in CSV")

    # Normalize coordinates to [0,1]
    all_points = np.concatenate([traj for traj in trajectories])
    lon_min, lat_min = all_points.min(axis=0)
    lon_max, lat_max = all_points.max(axis=0)

    # Avoid division by zero
    lon_range = lon_max - lon_min if lon_max != lon_min else 1.0
    lat_range = lat_max - lat_min if lat_max != lat_min else 1.0

    normalized_trajectories = []
    for traj in trajectories:
        normalized_traj = [
            ((lon - lon_min) / lon_range, (lat - lat_min) / lat_range)
            for lon, lat in traj
        ]
        normalized_trajectories.append(normalized_traj)

    return normalized_trajectories


# Example usage with the evaluation function
def evaluate_from_csv(real_csv_file, gen_csv_file, grid_size=16, length_bins=20):
    """
    Load real and generated trajectories from CSV and evaluate.

    Parameters:
    - real_csv_file: Path to real trajectories CSV.
    - gen_csv_file: Path to generated trajectories CSV.
    - grid_size: Size of the grid (default=16).
    - length_bins: Number of bins for length histogram (default=20).

    Returns:
    - Metrics dictionary from evaluate_difftraj.
    """
    real_trajectories = load_and_preprocess_trajectories(real_csv_file)
    gen_trajectories = load_and_preprocess_trajectories(gen_csv_file)
    metrics = evaluate_difftraj(real_trajectories, gen_trajectories, grid_size, length_bins)
    return metrics


import pandas as pd
import numpy as np


def get_real_bounds(real_csv_file):
    """
    Load the min and max lon/lat from real trajectories CSV.

    Parameters:
    - real_csv_file: Path to real trajectories CSV.

    Returns:
    - bounds: Tuple (lon_min, lon_max, lat_min, lat_max).
    """
    df = pd.read_csv(real_csv_file)
    if not {'lon', 'lat'}.issubset(df.columns):
        raise ValueError("CSV must contain 'lon', 'lat' columns")
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    return lon_min, lon_max, lat_min, lat_max


def clip_and_segment_trajectory(traj, lon_min, lon_max, lat_min, lat_max, min_points=2):
    """
    Clip trajectory points to bounds and segment into valid sub-trajectories.

    Parameters:
    - traj: NumPy array of shape (n_points, 2) with [lon, lat].
    - lon_min, lon_max, lat_min, lat_max: Bounds from real trajectories.
    - min_points: Minimum number of points for a valid sub-trajectory.

    Returns:
    - segments: List of NumPy arrays, each a valid sub-trajectory.
    """
    segments = []
    current_segment = []

    for point in traj:
        lon, lat = point
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            current_segment.append(point)
        else:
            if len(current_segment) >= min_points:
                segments.append(np.array(current_segment))
            current_segment = []

    if len(current_segment) >= min_points:
        segments.append(np.array(current_segment))

    return segments


def clip_generated_trajectories(real_csv_file, gen_csv_file, output_file='clipped_generated_trajectories.csv'):
    """
    Clip generated trajectories to real trajectory bounds and save to CSV.

    Parameters:
    - real_csv_file: Path to real trajectories CSV (for bounds).
    - gen_csv_file: Path to generated trajectories CSV (to be clipped).
    - output_file: Path to save the clipped trajectories CSV.
    """
    # Load bounds from real trajectories
    lon_min, lon_max, lat_min, lat_max = get_real_bounds(real_csv_file)
    print(f"Real bounds: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

    # Load generated trajectories
    df = pd.read_csv(gen_csv_file)
    if not {'id', 'lon', 'lat'}.issubset(df.columns):
        raise ValueError("Generated CSV must contain 'id', 'lon', 'lat' columns")

    # Group by 'id' to get trajectories
    trajectories = []
    for traj_id, group in df.groupby('id'):
        points = group[['lon', 'lat']].values
        if len(points) >= 2:  # Ensure trajectory has at least 2 points
            trajectories.append(points)

    # Clip and segment trajectories
    clipped_trajectories = []
    for traj in trajectories:
        segments = clip_and_segment_trajectory(traj, lon_min, lon_max, lat_min, lat_max)
        clipped_trajectories.extend(segments)

    if not clipped_trajectories:
        print("Warning: No valid trajectories after clipping")
        return

    # Save to CSV
    data = []
    for traj_id, traj in enumerate(clipped_trajectories, 1):  # Start ID from 1
        for lon, lat in traj:
            data.append([traj_id, lon, lat])

    clipped_df = pd.DataFrame(data, columns=['id', 'lon', 'lat'])
    clipped_df.to_csv(output_file, index=False)
    print(f"Clipped trajectories saved to {output_file} with {len(clipped_trajectories)} valid segments")


if __name__ == '__main__':
    # Example usage
    real_csv_file = './chengdu/dataset/save/limit_traj.csv'
    gen_path = R"D:\MyProjects\PythonAbout\DiffusionModel\MyDiffTraj\DataProcessing\ChengDu\05-13-16-25-30"
    gen_csv_file = gen_path+"\\generated_trajectories.csv"
    output_gen_file = gen_path+'\\clipped_generated_trajectories.csv'
    clip_generated_trajectories(real_csv_file, gen_csv_file,output_gen_file)
    metrics = evaluate_from_csv(real_csv_file, output_gen_file)
    plot_2d_csv(traj_2d_csv=output_gen_file,save_img_dir=gen_path+'\\',
                img_name='DiffTraj')
    print(metrics)
