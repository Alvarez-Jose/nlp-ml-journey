import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_json("train.json")
val_df = pd.read_json("val.json")

with open('file_contents.json', 'r') as f:
    file_contents = json.load(f)

# Basic statistics
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Number of source files: {len(file_contents)}")
print(f"Unique tags: {train_df['tag'].unique()}")

# Tag distribution
tag_dist = train_df['tag'].value_counts()
print("\nTag distribution:")
print(tag_dist)

# Annotation length analysis
train_df['annotation_length'] = train_df['end'] - train_df['start']
print(f"\nAvg annotation length: {train_df['annotation_length'].mean():.2f}")
print(f"Max annotation length: {train_df['annotation_length'].max()}")

# Visualize tag distribution
plt.figure(figsize=(10, 6))
tag_dist.plot(kind='bar')
plt.title('Tag Distribution in Training Data')
plt.xlabel('Tags')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('tag_distribution.png')
plt.show()