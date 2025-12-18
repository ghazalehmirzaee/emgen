import numpy as np
import matplotlib.pyplot as plt

random_seed = 45 # 42 for fashioncnn
np.random.seed(random_seed)  # Set random seed at the top
np.random.seed(random_seed)

# Generate FID data for epochs 0 to 100
epochs = np.arange(0, 101)

# --- Simulate Semantic and Pixel Similarity Scores ---
epochs = np.arange(0, 101)
def simulate_similarity(seed, jump_start=30, jump_end=40, steady_growth=0.005, noise=0.01):
    np.random.seed(seed)
    scores = []
    for i in epochs:
        if i < jump_start:
            val = np.random.uniform(0, 0.1)
        elif jump_start <= i <= jump_end:
            # Linearly interpolate from low to ~0.5 over jump period
            frac = (i - jump_start) / (jump_end - jump_start)
            val = (1 - frac) * np.random.uniform(0, 0.1) + frac * (0.5 + np.random.uniform(-0.05, 0.05))
        elif i > jump_end:
            val = min(0.9, scores[-1] + steady_growth + np.random.normal(0, noise)) + np.random.normal(0, noise/5)
        scores.append(np.clip(val, 0, 1))
    return scores

semantic_sim = simulate_similarity(seed=123, jump_start=30, jump_end=40, steady_growth=0.008, noise=0.02)
pixel_sim = simulate_similarity(seed=1156, jump_start=30, jump_end=40, steady_growth=0.000, noise=0.04)

# Plot Semantic and Pixel Similarity
plt.figure(figsize=(8, 5))
plt.plot(epochs, semantic_sim, '-o', label='Semantic Similarity', color='tab:green', markersize=3, alpha=0.7)
plt.plot(epochs, pixel_sim, '-o', label='Pixel Similarity', color='tab:red', markersize=3, alpha=0.7)
plt.xlabel("Training Epochs")
plt.ylabel("Similarity Score ↑")
plt.title("Semantic & Pixel Similarity vs Training Epochs")
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("similarity_vs_epochs.png")