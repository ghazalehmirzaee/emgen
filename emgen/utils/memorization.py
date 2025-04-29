import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import os

log = logging.getLogger(__name__)


def l2_distance(x, y):
    """
    Compute L2 (Euclidean) distance between samples.

    Args:
        x (torch.Tensor or np.ndarray): First sample/batch
        y (torch.Tensor or np.ndarray): Second sample/batch

    Returns:
        Distance(s) between samples
    """
    if isinstance(x, torch.Tensor):
        return torch.norm(x.view(x.size(0), -1) - y.view(y.size(0), -1), dim=1)
    else:
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        return np.linalg.norm(x_flat - y_flat, axis=1)


def compute_ssim(x, y):
    """
    Compute SSIM between two images or batches.

    Args:
        x (torch.Tensor or np.ndarray): First image/batch
        y (torch.Tensor or np.ndarray): Second image/batch

    Returns:
        SSIM value(s)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # Handle different dimensions
    if len(x.shape) == 2:  # 2D data
        return ssim(x, y, data_range=x.max() - x.min())
    elif len(x.shape) == 3 and x.shape[0] == 1:  # Single channel image
        return ssim(x[0], y[0], data_range=x.max() - x.min())
    elif len(x.shape) == 3 and x.shape[0] == 3:  # RGB image
        # Average SSIM across channels
        return np.mean([ssim(x[c], y[c], data_range=x.max() - x.min()) for c in range(3)])
    elif len(x.shape) == 4:  # Batch of images
        ssim_values = []
        for i in range(x.shape[0]):
            if x.shape[1] == 1:  # Single channel
                ssim_values.append(ssim(x[i, 0], y[i, 0], data_range=x.max() - x.min()))
            else:  # Multi-channel
                channel_ssims = [ssim(x[i, c], y[i, c], data_range=x.max() - x.min())
                                 for c in range(x.shape[1])]
                ssim_values.append(np.mean(channel_ssims))
        return np.array(ssim_values)
    else:
        raise ValueError(f"Unsupported shapes: {x.shape} and {y.shape}")


def compute_memorization_l2(generated_samples, training_samples, k=50, alpha=0.5):
    """
    Compute L2-based memorization metric.

    Args:
        generated_samples: Generated samples [B, ...]
        training_samples: Training samples [N, ...]
        k: Number of nearest neighbors
        alpha: Scaling constant

    Returns:
        Dictionary with memorization scores and related data
    """
    # Convert to numpy if tensors
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()
    if isinstance(training_samples, torch.Tensor):
        training_samples = training_samples.detach().cpu().numpy()

    # Flatten samples for nearest neighbor search
    generated_flat = generated_samples.reshape(generated_samples.shape[0], -1)
    training_flat = training_samples.reshape(training_samples.shape[0], -1)

    # Find k+1 nearest neighbors
    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(training_flat)
    distances, indices = nn_model.kneighbors(generated_flat)

    results = {
        'memorization_scores': [],
        'nearest_neighbors': [],
        'nearest_distances': [],
        'avg_distances': []
    }

    # Calculate memorization score for each sample
    for i in range(generated_samples.shape[0]):
        nearest_idx = indices[i, 0]
        nearest_distance = distances[i, 0]

        # Average distance to k nearest neighbors
        avg_distance = np.mean(distances[i, 1:k + 1])

        # Compute memorization score (negative ratio of distances)
        memorization_score = -nearest_distance / (alpha * avg_distance)

        results['memorization_scores'].append(memorization_score)
        results['nearest_neighbors'].append(nearest_idx)
        results['nearest_distances'].append(nearest_distance)
        results['avg_distances'].append(avg_distance)

    return results


def compute_memorization_ssim(generated_samples, training_samples, k=50, alpha=0.5):
    """
    Compute SSIM-based memorization metric.

    Args:
        generated_samples: Generated samples [B, ...]
        training_samples: Training samples [N, ...]
        k: Number of nearest neighbors
        alpha: Scaling constant

    Returns:
        Dictionary with memorization scores and related data
    """
    # Convert to numpy if tensors
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()
    if isinstance(training_samples, torch.Tensor):
        training_samples = training_samples.detach().cpu().numpy()

    results = {
        'memorization_scores': [],
        'nearest_neighbors': [],
        'nearest_similarities': [],
        'avg_similarities': []
    }

    # For each generated sample
    for i in range(generated_samples.shape[0]):
        similarities = []

        # Compute similarity to all training samples
        for j in range(training_samples.shape[0]):
            sim = compute_ssim(
                generated_samples[i:i + 1],
                training_samples[j:j + 1]
            )
            similarities.append((sim, j))

        # Sort by similarity (higher is more similar for SSIM)
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Get nearest neighbor and its similarity
        nearest_similarity, nearest_idx = similarities[0]

        # Get average similarity to k nearest neighbors
        avg_similarity = np.mean([sim for sim, _ in similarities[1:k + 1]])

        # Compute memorization score
        # Note: For SSIM, higher values mean more similar, so we invert the ratio
        memorization_score = -nearest_similarity / (alpha * avg_similarity)

        results['memorization_scores'].append(memorization_score)
        results['nearest_neighbors'].append(nearest_idx)
        results['nearest_similarities'].append(nearest_similarity)
        results['avg_similarities'].append(avg_similarity)

    return results


def plot_memorization_results(generated_samples, training_samples, results, metric_type, save_dir, top_n=10):
    """
    Visualize memorization results.

    Args:
        generated_samples: Generated samples
        training_samples: Training samples
        results: Results from memorization computation
        metric_type: Type of metric ('l2' or 'ssim')
        save_dir: Directory to save plots
        top_n: Number of top cases to plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot distribution of memorization scores
    plt.figure(figsize=(10, 6))
    plt.hist(results['memorization_scores'], bins=50)
    plt.title(f"Distribution of {metric_type.upper()} Memorization Scores")
    plt.xlabel("Memorization Score")
    plt.ylabel("Count")
    plt.axvline(x=0, color='r', linestyle='--',
                label="Threshold (scores > 0 may indicate memorization)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{metric_type}_memorization_histogram.png")
    plt.close()

    # Sort by memorization score
    sorted_indices = np.argsort(results['memorization_scores'])
    most_memorized = sorted_indices[-top_n:][::-1]  # Highest scores = most memorization
    least_memorized = sorted_indices[:top_n]  # Lowest scores = least memorization

    # Visualize differently based on data type
    is_image_data = len(generated_samples.shape) > 2

    if is_image_data:
        _plot_image_memorization(generated_samples, training_samples, results,
                                 most_memorized, "highest", metric_type, save_dir)
        _plot_image_memorization(generated_samples, training_samples, results,
                                 least_memorized, "lowest", metric_type, save_dir)
    else:
        _plot_2d_memorization(generated_samples, training_samples, results,
                              most_memorized, "highest", metric_type, save_dir)
        _plot_2d_memorization(generated_samples, training_samples, results,
                              least_memorized, "lowest", metric_type, save_dir)


def _plot_image_memorization(generated_samples, training_samples, results, indices, label, metric_type, save_dir):
    """Helper function to plot image memorization results"""
    n_cols = 4  # (generated, nearest neighbor, difference, score)
    n_rows = min(len(indices), 10)

    plt.figure(figsize=(12, 3 * n_rows))

    for i, idx in enumerate(indices[:n_rows]):
        # Convert index to integer
        idx_int = int(idx)
        nn_idx = int(results['nearest_neighbors'][idx_int])

        # Get samples
        gen_sample = generated_samples[idx_int]
        train_sample = training_samples[nn_idx]

        # Convert score to a scalar float
        score = float(results['memorization_scores'][idx_int])

        # Convert to channel-last format if needed
        if len(gen_sample.shape) > 2 and gen_sample.shape[0] in [1, 3]:
            gen_sample = np.transpose(gen_sample, (1, 2, 0))
            train_sample = np.transpose(train_sample, (1, 2, 0))

        # Squeeze single channels
        if len(gen_sample.shape) > 2 and gen_sample.shape[-1] == 1:
            gen_sample = gen_sample.squeeze(-1)
            train_sample = train_sample.squeeze(-1)

        # Calculate difference
        diff = np.abs(gen_sample - train_sample)

        # Plot generated sample
        plt.subplot(n_rows, n_cols, i * n_cols + 1)
        plt.imshow(gen_sample, cmap='gray' if len(gen_sample.shape) == 2 else None)
        plt.title("Generated")
        plt.axis('off')

        # Plot nearest neighbor
        plt.subplot(n_rows, n_cols, i * n_cols + 2)
        plt.imshow(train_sample, cmap='gray' if len(train_sample.shape) == 2 else None)
        plt.title("Nearest Neighbor")
        plt.axis('off')

        # Plot difference
        plt.subplot(n_rows, n_cols, i * n_cols + 3)
        plt.imshow(diff, cmap='hot')
        plt.title("Difference")
        plt.axis('off')

        # Plot score - with proper scalar formatting
        plt.subplot(n_rows, n_cols, i * n_cols + 4)
        plt.text(0.5, 0.5, f"Score: {score:.2f}",
                 ha='center', va='center',
                 transform=plt.gca().transAxes,
                 fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / f"{metric_type}_{label}_memorization.png")
    plt.close()



def _plot_2d_memorization(generated_samples, training_samples, results, indices, label, metric_type, save_dir):
    """Helper function to plot 2D memorization results"""
    plt.figure(figsize=(12, 10))
    plt.scatter(training_samples[:, 0], training_samples[:, 1], s=1, alpha=0.1,
                color='gray', label="Training Data")

    for i, idx in enumerate(indices):
        # Convert index to integer
        idx_int = int(idx)
        nn_idx = int(results['nearest_neighbors'][idx_int])

        # Get samples
        gen_sample = generated_samples[idx_int]
        train_sample = training_samples[nn_idx]

        # Convert score to a scalar float
        score = float(results['memorization_scores'][idx_int])

        # Plot generated sample and its nearest neighbor
        plt.scatter(gen_sample[0], gen_sample[1], color='red', s=100,
                    label="Generated" if i == 0 else None)
        plt.scatter(train_sample[0], train_sample[1], color='blue', s=100,
                    label="Nearest Neighbor" if i == 0 else None)

        # Connect with a line
        plt.plot([gen_sample[0], train_sample[0]],
                 [gen_sample[1], train_sample[1]], 'k--', alpha=0.7)

        # Add score label with proper scalar formatting
        plt.annotate(f"{score:.2f}",
                     xy=((gen_sample[0] + train_sample[0]) / 2,
                         (gen_sample[1] + train_sample[1]) / 2),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=9)

    plt.title(f"{label.capitalize()} {metric_type.upper()} Memorization Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{metric_type}_{label}_memorization_2d.png")
    plt.close()

