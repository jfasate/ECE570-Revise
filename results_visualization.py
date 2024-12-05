import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
results_dir = '/home/jayesh/Documents/CNN/results/'
k_scores = np.load(f'{results_dir}k_scores.npy')
f1_scores = np.load(f'{results_dir}f1_scores.npy')
p_scores = np.load(f'{results_dir}p_scores.npy')
r_scores = np.load(f'{results_dir}r_scores.npy')
losses = np.load(f'{results_dir}losses.npy')
oa_scores = np.load(f'{results_dir}oa_arr.npy')

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Box plot of Kappa scores
ax1.boxplot(k_scores)
ax1.set_title('Kappa Scores Distribution')
ax1.set_ylabel('Score')
ax1.grid(True)

# Plot 2: Learning curve (losses)
ax2.plot(losses, '-')
ax2.set_title('Training Loss')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.grid(True)

# Plot 3: F1, Precision, Recall scores per class
class_metrics = pd.DataFrame({
    'F1': f1_scores.mean(axis=0),
    'Precision': p_scores.mean(axis=0),
    'Recall': r_scores.mean(axis=0)
})
class_metrics.plot(kind='bar', ax=ax3)
ax3.set_title('Performance Metrics by Class')
ax3.set_xlabel('Class')
ax3.set_ylabel('Score')
ax3.legend()
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Plot 4: Overall Accuracy distribution
ax4.boxplot(oa_scores)
ax4.set_title('Overall Accuracy Distribution')
ax4.set_ylabel('Accuracy')
ax4.grid(True)

plt.tight_layout()
plt.savefig(f'{results_dir}model_performance.png', dpi=300, bbox_inches='tight')
plt.close()