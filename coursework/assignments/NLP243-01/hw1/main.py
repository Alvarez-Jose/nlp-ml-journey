"""
Main script for training a Bag-of-Words classifier.

This script loads data, sets up the model, and trains it using the specified
training and validation datasets.
"""

import argparse
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocess import get_data
from model import BoWClassifier, predict
from train import train_model


def parse_arguments():
    """Parse command line arguments for training the classifier."""
    parser = argparse.ArgumentParser(description='Train a Bag-of-Words classifier')
    parser.add_argument('--train_path', default='./data/train.csv', help='Path to training data CSV file')
    parser.add_argument('--val_path', default='./data/val.csv', help='Path to validation data CSV file')
    parser.add_argument('--test_path', default='./data/test.csv', help='Path to test data CSV file')
    parser.add_argument('--labels_path', default='./data/all_labels.csv', help='Path to labels CSV file')
    parser.add_argument('--vectorizer_path', default='./vectorizer.joblib', help='Path to saved vectorizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')  
    return parser.parse_args()


def load_labels_and_vectorizer(args):
    """Load labels and create label mapping, then load the vectorizer."""
    # Load labels and create label mapping
    with open(args.labels_path) as f_labels:
        # filter out empty lines and a possible header like "label"
        labels = [ln.strip() for ln in f_labels if ln.strip() and ln.strip().lower() != 'label']
    label_to_id = {label: i for i, label in enumerate(labels)}
    
    # Load pretrained vectorizer
    bow_vectorizer = joblib.load(args.vectorizer_path)
    
    return label_to_id, bow_vectorizer


def prepare_data_loaders(args, bow_vectorizer, label_to_id, batch_size=64):
    """Load training, validation, and test data and create data loaders."""
    # Load training and validation data
    x_train, y_train = get_data(args.train_path, bow_vectorizer,
                                include_y=True, label_to_id=label_to_id)
    x_val, y_val = get_data(args.val_path, bow_vectorizer,
                            include_y=True, label_to_id=label_to_id)
    x_test, y_test = get_data(args.test_path, bow_vectorizer,
                              include_y=True, label_to_id=label_to_id)

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset   = TensorDataset(x_val,   y_val)
    test_dataset  = TensorDataset(x_test,  y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Main function to train the BoW classifier."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load labels and vectorizer
    label_to_id, bow_vectorizer = load_labels_and_vectorizer(args)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        args, bow_vectorizer, label_to_id, batch_size=args.batch_size
    )
    
    # Initialize model
    input_size = len(bow_vectorizer.vocabulary_)
    num_labels = len(label_to_id)
    bow_model = BoWClassifier(input_size=input_size, num_labels=num_labels)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = train_model(bow_model, predict, train_loader, val_loader, test_loader, device)
    
    # TODO: Repeat with different hyperparameters


if __name__ == "__main__":
    main()


