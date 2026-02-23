import argparse
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn import metrics

from tagger import load_treebank_splits, get_token_tag_tuples, evaluate, HMMTagger


class MostFrequentTagger:
    """
    Baseline: assign each word its most frequent tag in training.
    OOV words -> global most frequent tag.
    """
    def __init__(self):
        self.word2tag = {}
        self.default_tag = "NN"

    def fit(self, train_sents):
        word_tag_counts = defaultdict(Counter)
        tag_counts = Counter()

        for sent in train_sents:
            for w, t in sent:
                word_tag_counts[w][t] += 1
                tag_counts[t] += 1

        if tag_counts:
            self.default_tag = tag_counts.most_common(1)[0][0]
        self.word2tag = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}

    def tag_sentence(self, words):
        return [(w, self.word2tag.get(w, self.default_tag)) for w in words]


def tag_split(model, sents):
    return [model.tag_sentence([w for w, _ in sent]) for sent in sents]


def accuracy(gold_sents, pred_sents):
    gold = [t for sent in gold_sents for _, t in sent]
    pred = [t for sent in pred_sents for _, t in sent]
    correct = sum(g == p for g, p in zip(gold, pred))
    return correct / max(1, len(gold))


def tune_alpha(train_sents, dev_sents, alphas):
    best_alpha = alphas[0]
    best_acc = -1.0

    print("\nTuning alpha on dev...")
    for a in alphas:
        hmm = HMMTagger(alpha=a)
        hmm.fit(train_sents)
        pred_dev = tag_split(hmm, dev_sents)
        acc = accuracy(dev_sents, pred_dev)
        print(f"  alpha={a:<6} dev_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_alpha = a

    print(f"Best alpha: {best_alpha} (dev_acc={best_acc:.4f})")
    return best_alpha


def confusion_and_top_confusions(gold_sents, pred_sents, labels, topk=25):
    gold = [str(t) for sent in gold_sents for _, t in sent]
    pred = [str(t) for sent in pred_sents for _, t in sent]

    label_set = set(labels)
    pair_counts = defaultdict(int)

    for g, p in zip(gold, pred):
        if g in label_set and p in label_set and g != p:
            pair_counts[(g, p)] += 1

    top = sorted(((c, g, p) for (g, p), c in pair_counts.items()), reverse=True)[:topk]
    return None, top



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str,
        default=os.path.join("data", "penn-treebank3-wsj", "wsj"),
        help="Path to .../wsj directory with section folders 00..24",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.1,0.25,0.5,1,2,5")
    args = parser.parse_args()

    train_raw, dev_raw, test_raw = load_treebank_splits(args.datadir)

    train_sents = [get_token_tag_tuples(s) for s in train_raw]
    dev_sents = [get_token_tag_tuples(s) for s in dev_raw]
    test_sents = [get_token_tag_tuples(s) for s in test_raw]

    # tag labels (train tagset)
    labels = sorted({t for sent in train_sents for _, t in sent if isinstance(t, str) and t})



    # 1) Baseline
    baseline = MostFrequentTagger()
    baseline.fit(train_sents)

    pred_dev_base = tag_split(baseline, dev_sents)
    pred_test_base = tag_split(baseline, test_sents)

    print(f"\nBaseline accuracy: dev={accuracy(dev_sents, pred_dev_base):.4f}  test={accuracy(test_sents, pred_test_base):.4f}")
    evaluate(test_sents, pred_test_base)

    # 2) Tune alpha on dev (optional)
    alpha = args.alpha
    if args.tune:
        alpha_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
        alpha = tune_alpha(train_sents, dev_sents, alpha_list)

    # 3) HMM
    hmm = HMMTagger(alpha=alpha)
    print(f"\nTraining HMM (alpha={alpha})...")
    hmm.fit(train_sents)

    pred_dev_hmm = tag_split(hmm, dev_sents)
    pred_test_hmm = tag_split(hmm, test_sents)

    print(f"\nHMM accuracy: dev={accuracy(dev_sents, pred_dev_hmm):.4f}  test={accuracy(test_sents, pred_test_hmm):.4f}")
    evaluate(test_sents, pred_test_hmm)

    # 4) Confusion matrix + top confusions
    cm, top = confusion_and_top_confusions(test_sents, pred_test_hmm, labels=labels, topk=25)
    print("\nTop confusions (gold -> predicted : count):")
    for c, g, p in top:
        print(f"  {g:>6} -> {p:<6} : {c}")


if __name__ == "__main__":
    main()
