import os
import math
import collections
import nltk
from sklearn import metrics


def evaluate(test_sentences, tagged_test_sentences):
    gold = [str(tag) for sentence in test_sentences for token, tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences for token, tag in sentence]
    print(metrics.classification_report(gold, pred, zero_division=0))


def get_token_tag_tuples(sent):
    pairs = []
    for tok in sent.split():
        w, t = nltk.tag.str2tuple(tok)
        if w is None or t is None:
            continue
        pairs.append((w, t))
    return pairs


def get_tagged_sentences(text):
    sentences = []
    blocks = text.split("============================")

    for block in blocks:
        sents = block.split("\n\n")
        for sent in sents:
            sent = sent.replace("\n", "").replace("[", "").replace("]", "").strip()
            if sent:
                sentences.append(sent)

    return sentences


def load_treebank_splits(datadir):
    train, dev, test = [], [], []

    print("Loading treebank data from:", datadir)
    if not os.path.isdir(datadir):
        raise FileNotFoundError(f"datadir does not exist: {datadir}")

    for subdir, _, files in os.walk(datadir):
        for filename in files:
            if not filename.endswith(".pos"):
                continue

            section_str = os.path.basename(subdir)
            if not section_str.isdigit():
                continue
            section = int(section_str)

            filepath = os.path.join(subdir, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()

            if section in range(0, 19):
                train += get_tagged_sentences(text)
            elif section in range(19, 22):
                dev += get_tagged_sentences(text)
            elif section in range(22, 25):
                test += get_tagged_sentences(text)

    print("Train set size:", len(train))
    print("Dev set size:", len(dev))
    print("Test set size:", len(test))
    return train, dev, test


class HMMTagger:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.start_tag = "<START>"
        self.stop_tag = "<STOP>"
        self.unk_token = "<UNK>"

        self.tags = []
        self.vocab = set()
        self.log_trans = {}
        self.log_emit = {}

    def fit(self, train_sents):
        tagset = set()
        vocab = set()

        for sent in train_sents:
            for word, tag in sent:
                if word is None or tag is None:
                    continue
                vocab.add(word)
                tagset.add(tag)

        vocab.add(self.unk_token)

        self.tags = sorted(tagset)
        self.vocab = set(vocab)

        trans_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        emit_counts = collections.defaultdict(lambda: collections.defaultdict(int))

        for sent in train_sents:
            prev_tag = self.start_tag
            for word, tag in sent:
                if word is None or tag is None:
                    continue
                trans_counts[prev_tag][tag] += 1
                emit_counts[tag][word] += 1
                prev_tag = tag
            trans_counts[prev_tag][self.stop_tag] += 1

        self.log_trans = {}
        possible_next = self.tags + [self.stop_tag]
        num_next = len(possible_next)
        prev_tags = [self.start_tag] + self.tags

        for prev_tag in prev_tags:
            self.log_trans[prev_tag] = {}
            total = sum(trans_counts[prev_tag][nxt] for nxt in possible_next)
            denom = total + self.alpha * num_next

            for nxt in possible_next:
                count = trans_counts[prev_tag][nxt]
                prob = (count + self.alpha) / denom
                self.log_trans[prev_tag][nxt] = math.log(prob)

        self.log_emit = {}
        vocab_list = list(self.vocab)
        vocab_size = len(vocab_list)

        for tag in self.tags:
            self.log_emit[tag] = {}
            total = sum(emit_counts[tag].values())
            denom = total + self.alpha * vocab_size

            for word in vocab_list:
                count = emit_counts[tag][word]
                prob = (count + self.alpha) / denom
                self.log_emit[tag][word] = math.log(prob)

    def _emit_logp(self, tag, word):
        w = word if word in self.vocab else self.unk_token
        return self.log_emit[tag][w]

    def _viterbi(self, words):
        if not words:
            return []

        n = len(words)
        dp = [dict() for _ in range(n)]
        backptr = [dict() for _ in range(n)]

        w0 = words[0]
        for tag in self.tags:
            dp[0][tag] = self.log_trans[self.start_tag][tag] + self._emit_logp(tag, w0)
            backptr[0][tag] = self.start_tag

        for j in range(1, n):
            wj = words[j]
            for curr_tag in self.tags:
                best_score = float("-inf")
                best_prev = None
                emit = self._emit_logp(curr_tag, wj)

                for prev_tag in self.tags:
                    score = dp[j - 1][prev_tag] + self.log_trans[prev_tag][curr_tag] + emit
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag

                dp[j][curr_tag] = best_score
                backptr[j][curr_tag] = best_prev

        best_last = None
        best_final = float("-inf")
        for last_tag in self.tags:
            score = dp[n - 1][last_tag] + self.log_trans[last_tag][self.stop_tag]
            if score > best_final:
                best_final = score
                best_last = last_tag

        best_tags = [None] * n
        best_tags[-1] = best_last
        for j in range(n - 1, 0, -1):
            best_tags[j - 1] = backptr[j][best_tags[j]]

        return best_tags

    def tag_sentence(self, words):
        tags = self._viterbi(words)
        return list(zip(words, tags))
