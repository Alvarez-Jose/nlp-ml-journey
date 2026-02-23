import pandas as pd
import json
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_sequence_data(tf, file_contents, max_seq_length=512):
    sequences = []
    labels = [] 

    # Find group annotations by file
    for fileid in df['fileid'].unique():
        file_df = df[df['fileid'] == fileid]
        text = file_contents[fileid]

        # Initialize BIO tags
        bio_tags = ['0'] * len(text)

        # Now assign BIO tags based on annotations
        for _, row in file_df.iterrows():
            tag = row['tag']
            start, end = row['start'], row['end']

            # B- tag for first chara, i- for rest
            bio_tags[start] = f'B-{tag}'
            for i in range(start + 1, end):
                bio_tags[i] = f'I-{tag}'

        # split into sentences using tokenization
        for i in range(0, len(text), max_seq_length):
            seq = text[i:i + max_seq_length]
            seq_labels = bio_tags[i:i + max_seq_length]

            if len(seq) > 0:
                sequences.append(seq)
                labels.append(seq_labels)

    return sequences, labels

# load the data
train_df = pd.read_json("train.json")
val_df = pd.read_json("validation.json")

with open('file_contents.json', 'r') as f:
    file_contents = json.load(f)

# prepare sequences
train_sequences, train_labels = prepare_sequence_data(train_df, file_contents)
val_sequences, val_labels = prepare_sequence_data(val_df, file_contents)

# Save preprocessed data
np.save('train_sequences.npy', np.array(train_sequences, dtype=object))
np.save('train_labels.npy', np.array(train_labels, dtype=object))
np.save('val_sequences.npy', np.array(val_sequences, dtype=object))
np.save('val_labels.npy', np.array(val_labels, dtype=object))

print(f"Training sequences: {len(train_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")