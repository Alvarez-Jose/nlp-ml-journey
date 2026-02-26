import pandas as pd
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions and ground truth (if available)
with open('all_predictions.json', 'r') as f:
    predictions = json.load(f)

# Load validation data for error analysis
val_df = pd.read_json("val.json")

# Analyze prediction distribution
all_tags = []
for filename, preds in predictions.items():
    for pred in preds:
        all_tags.append(pred['tag'])

tag_counts = Counter(all_tags)
print("Predicted tag distribution:")
for tag, count in tag_counts.most_common():
    print(f"  {tag}: {count}")

# Compare with ground truth distribution
truth_tags = val_df['tag'].tolist()
truth_counts = Counter(truth_tags)

print("\nGround truth tag distribution:")
for tag, count in truth_counts.most_common():
    print(f"  {tag}: {count}")

# Error analysis by file
print("\nPredictions per file:")
for filename, preds in predictions.items():
    print(f"  {filename}: {len(preds)} predictions")

# Confidence analysis (if you have confidence scores)
# Add confidence scores to predictions in inference step first

# Visualize tag distribution comparison
tags = sorted(set(list(tag_counts.keys()) + list(truth_counts.keys())))
pred_values = [tag_counts.get(tag, 0) for tag in tags]
truth_values = [truth_counts.get(tag, 0) for tag in tags]

x = range(len(tags))
plt.figure(figsize=(12, 6))
plt.bar(x, truth_values, width=0.4, label='Ground Truth', alpha=0.7)
plt.bar([i + 0.4 for i in x], pred_values, width=0.4, label='Predictions', alpha=0.7)
plt.xlabel('Tags')
plt.ylabel('Count')
plt.title('Tag Distribution: Ground Truth vs Predictions')
plt.xticks([i + 0.2 for i in x], tags, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('error_analysis_distribution.png')
plt.show()

# Save error analysis summary
error_summary = {
    'total_predictions': len(all_tags),
    'tag_distribution': dict(tag_counts),
    'ground_truth_distribution': dict(truth_counts),
    'predictions_per_file': {f: len(p) for f, p in predictions.items()}
}

with open('error_analysis_summary.json', 'w') as f:
    json.dump(error_summary, f, indent=2)

print("\nError analysis complete!")