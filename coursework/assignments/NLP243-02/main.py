from data import GPTTokenizedData


def main():
    # get dataloaders (data.py)
    tokenized = GPTTokenizedData()
    dataloaders = tokenized.dataloaders # all 3 dataloaders in a dictionary with keys 'train', 'test', 'val
    vocab_size = tokenized.vocab_size


    # instantiate model (model.py)

    
    # train model (train.py)
    
    
    # evaluate perplexity for all three splits (evaluate.py)


if __name__ == "__main__":
    main()
