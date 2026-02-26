import torch
import os
import json
import glob
import numpy as np
from transformers import AutoTokenizer
import pandas as pd

from model import NERModel

def predict_file(model, tokenizer, file_path, device, label2id, id2label, max_length=512):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"  Reading file: {len(text)} characters")
    
    # Split text into chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    print(f"  Created {len(chunks)} chunks")
    
    all_entities = []
    
    model.eval()
    with torch.no_grad():
        for chunk_idx, chunk in enumerate(chunks):
            # Tokenize
            encoding = tokenizer(
                chunk,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get predictions
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # Convert token predictions to entities
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            word_ids = encoding.word_ids()
            
            current_entity = None
            current_text = []
            current_start = None
            entity_count = 0
            
            for i, (pred, word_idx) in enumerate(zip(preds, word_ids)):
                if word_idx is None:
                    continue
                    
                label = id2label[pred]
                
                # FIX: Since model only predicts I- tags, treat any I- tag as entity start
                if label.startswith('I-'):
                    clean_label = label.replace('I-', '')
                    
                    # If this is a new entity (different type or no current entity)
                    if current_entity is None or clean_label != current_entity:
                        # Save previous entity if exists
                        if current_entity is not None:
                            entity_text = ' '.join(current_text).replace('##', '')
                            all_entities.append({
                                'start': current_start,
                                'end': i,
                                'tag': current_entity,
                                'text': entity_text
                            })
                            entity_count += 1
                            print(f"      Saved entity: {current_entity} - {entity_text}")
                        
                        # Start new entity
                        current_entity = clean_label
                        current_text = [tokens[i]]
                        current_start = i
                        print(f"      Started new entity: {current_entity} with token {tokens[i]}")
                    else:
                        # Continue current entity
                        current_text.append(tokens[i])
                
                # Outside tag
                elif label == 'O':
                    # Save previous entity if exists
                    if current_entity is not None:
                        entity_text = ' '.join(current_text).replace('##', '')
                        all_entities.append({
                            'start': current_start,
                            'end': i,
                            'tag': current_entity,
                            'text': entity_text
                        })
                        entity_count += 1
                        print(f"      Saved entity: {current_entity} - {entity_text}")
                        current_entity = None
                        current_text = []
                        current_start = None
            
            # Don't forget to add the last entity
            if current_entity is not None:
                entity_text = ' '.join(current_text).replace('##', '')
                all_entities.append({
                    'start': current_start,
                    'end': len(preds),
                    'tag': current_entity,
                    'text': entity_text
                })
                entity_count += 1
                print(f"      Saved final entity: {current_entity} - {entity_text}")
            
            print(f"      Found {entity_count} entities in chunk {chunk_idx+1}")
    
    print(f"  Total entities found: {len(all_entities)}")
    return all_entities

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label2id = np.load('label2id.npy', allow_pickle=True).item()
id2label = np.load('id2label.npy', allow_pickle=True).item()

model = NERModel(num_labels=len(label2id)).to(device)
model.load_state_dict(torch.load('ner_model.pt', map_location=device))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Run inference on all unannotated files
unannotated_files = glob.glob('./unannotated_mmds/*.mmd.filtered')

print(f"Found {len(unannotated_files)} unannotated files: {unannotated_files}")
results = {}

for file_path in unannotated_files:
    filename = os.path.basename(file_path)
    print(f"Processing {filename}...")
    
    predictions = predict_file(model, tokenizer, file_path, device, label2id, id2label)
    results[filename] = predictions
    
    # Save individual predictions
    df = pd.DataFrame(predictions)
    df.to_csv(f'predictions_{filename}.csv', index=False)

# Save all predictions
with open('all_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Inference complete!")