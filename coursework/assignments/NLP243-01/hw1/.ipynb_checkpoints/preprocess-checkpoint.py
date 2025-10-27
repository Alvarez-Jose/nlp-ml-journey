"""
Preprocessing utilities for NLP classification task.

This module provides functions for:
- Converting text data to bag-of-words representations
- Converting target labels to multi-hot encoded vectors
- Lemmatizing and tokenizing text data

You may edit this file if you prefer other preprocessing methods, 
as long as you follow the same interface (e.g. get_data, main).
"""

import csv
import joblib
from typing import List, Dict, Optional, Tuple, Union

import nltk
import torch
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


def target_to_manyhot(targets_ids: List[List[int]], n_ids: int) -> List[List[int]]:
    """
    Convert list of target IDs to multi-hot encoded vectors.
    
    Args:
        targets_ids: List of lists, where each inner list contains target IDs
        n_ids: Total number of possible IDs (determines vector length)
        
    Returns:
        List of multi-hot encoded vectors (list of lists of 0s and 1s)
        
    Example:
        >>> target_to_manyhot([[0, 2], [1]], 4)
        [[1, 0, 1, 0], [0, 1, 0, 0]]
    """
    result = []
    for target_ids in targets_ids:
        multi_hot = [0] * n_ids
        for idx in target_ids:
            multi_hot[idx] = 1
        result.append(multi_hot)
    return result 

def target_ids(targets: List[List[str]], rel_to_id: Dict[str, int]) -> List[List[int]]:
    """
    Convert relation names to their corresponding IDs.
    
    Args:
        targets: List of lists containing relation names
        rel_to_id: Dictionary mapping relation names to IDs
        
    Returns:
        List of lists containing relation IDs (excluding 'none' relations)
        
    Example:
        >>> target_ids([['cause', 'none', 'enable']], {'cause': 0, 'enable': 1})
        [[0, 1]]
    """
    result_ids = []
    for relations in targets:
        relation_ids = []
        for relation in relations:
            if relation == 'none':
                continue
            relation_ids.append(rel_to_id[relation])
        result_ids.append(relation_ids)
    return result_ids

def get_wordnet_pos(word: str) -> str:
    """
    Get the WordNet part-of-speech tag for a word.
    
    Args:
        word: The word to get POS tag for
        
    Returns:
        WordNet POS tag (ADJ, NOUN, VERB, ADV, or NOUN as default)
        
    Example:
        >>> get_wordnet_pos('running')
        'v'  # verb
        >>> get_wordnet_pos('beautiful')
        'a'  # adjective
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if no match

def lemmatized_sentence(sentence: str) -> str:
    """
    Lemmatize a sentence by reducing words to their base or dictionary form.
    
    This function:
    - Tokenizes the sentence into words and punctuation
    - Lemmatizes each word using its part-of-speech tag
    - Rejoins the lemmatized words into a sentence
    
    Args:
        sentence: Input sentence as a string
        
    Returns:
        Lemmatized sentence as a string
        
    Example:
        >>> lemmatized_sentence("The cats are running quickly.")
        "The cat be run quickly ."
        
    Note:
        Lemmatizer reduces words to base forms, e.g.:
        "running", "ran", "runs" -> "run"
        "better", "best" -> "good"
    """
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize sentence into words and punctuation
    words = nltk.word_tokenize(sentence)
    
    # Lemmatize each word using its POS tag
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word)) 
        for word in words
    ]
    
    return ' '.join(lemmatized_words)

def apply_lemmatizer(sentence_list: List[str]) -> List[str]:
    """
    Apply lemmatization to a list of sentences.
    
    Args:
        sentence_list: List of sentences to lemmatize
        
    Returns:
        List of lemmatized sentences
        
    Example:
        >>> apply_lemmatizer(["The cats run.", "Dogs are running."])
        ["The cat run .", "Dog be run ."]
    """
    return [lemmatized_sentence(sentence) for sentence in sentence_list]

def get_data(
    csv_path: str, 
    vectorizer: CountVectorizer, 
    include_y: bool = False, 
    label_to_id: Optional[Dict[str, int]] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load and preprocess data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing the data
        vectorizer: Fitted CountVectorizer for text-to-vector conversion
        include_y: Whether to include target labels in the output
        label_to_id: Dictionary mapping label names to IDs (required if include_y=True)
        
    Returns:
        If include_y=False: torch.Tensor of features (X)
        If include_y=True: Tuple of (features, targets) as torch.Tensors
        
    Raises:
        AssertionError: If include_y=True but label_to_id is None
        
    Example:
        >>> vectorizer = joblib.load('vectorizer.joblib')
        >>> X = get_data('data/train.csv', vectorizer, include_y=False)
        >>> X, y = get_data('data/train.csv', vectorizer, include_y=True, label_to_id=label_map)
    """
    # Load raw data from CSV
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        raw_data = list(reader)
    
    # Extract utterances (input text)
    raw_inputs = [row['UTTERANCES'] for row in raw_data]
    
    # Preprocess text: lemmatize and vectorize
    lemmatized_inputs = apply_lemmatizer(raw_inputs)
    features = torch.FloatTensor(vectorizer.transform(lemmatized_inputs).toarray())
    
    # Return only features if labels not requested
    if not include_y:
        return features
    
    # Process labels if requested
    assert label_to_id is not None, "label_to_id must be provided when include_y=True"
    
    raw_targets = [row['CORE RELATIONS'].split() for row in raw_data]
    target_id_lists = target_ids(raw_targets, label_to_id)
    targets = torch.FloatTensor(target_to_manyhot(target_id_lists, len(label_to_id)))
    
    return features, targets


def main():
    """
    Train and save a bag-of-words vectorizer using the training data.
    
    This function:
    1. Loads training data from data/train.csv
    2. Applies lemmatization to the text
    3. Fits a CountVectorizer with specified parameters
    4. Saves the trained vectorizer for later use
    
    The vectorizer is configured with:
    - max_df=0.95: Ignore terms that appear in more than 95% of documents
    - min_df=1: Include terms that appear in at least 1 document
    """
    train_path = './data/train.csv'
    vectorizer_path = 'vectorizer.joblib'
    
    print("Loading training data...")
    with open(train_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        raw_train_data = list(reader)
    
    # Extract and preprocess training utterances
    raw_inputs = [row['UTTERANCES'] for row in raw_train_data]
    lemmatized_inputs = apply_lemmatizer(raw_inputs)
    
    print(f"Training vectorizer on {len(lemmatized_inputs)} sentences...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=1)
    vectorizer.fit(lemmatized_inputs)
    
    # Save the trained vectorizer
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
