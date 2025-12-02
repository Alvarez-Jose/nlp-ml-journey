import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type = str)
    parser.add_argument("test_path", type = str)
    parser.add_argument("output_path", type = str)
    return parser.parse_args()

def dummy_pipeline(train_path: str, test_path: str, output_path: str):
    '''
    read train.csv
    read test.csv
    devlop a dummy test_pred.csv
    '''
    import csv
    
    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")
    print(f"Will save predictions to: {output_path}")
    
    # reads test csv 
    with open(test_path, newline = "", encoding = "utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} test examples")
    
    # creating a dummy predicitions file with all "O" tags
    with open(output_path, "W", newline = "", encoding = "utf-8") as f:
        fieldnames = ["id","labels"]
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            # assumes the sentences are spaced separated tokes
            sent = row["sentence"].strip().split()
            labels = "".join(["O"] * len(sent))
            writer.writerow({"id": i, "labels": labels})
    print("Wrote dummy predictions")
    
def main():
    args = parse_args()
    dummy_pipeline(args.train_path, args.test_path, args.output_path)
    
if __name__ == "__main__":
    main