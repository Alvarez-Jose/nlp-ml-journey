import os
import math
import collections
import nltk
from sklearn import metrics


def evaluate(test_sentences, tagged_test_sentences):
    gold = [str(tag) for sentence in test_sentences for token, tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences for token, tag in sentence]
    print(metrics.classification_report(gold, pred))


def get_token_tag_tuples(sent):
    return [nltk.tag.str2tuple(t) for t in sent.split()]


def get_tagged_sentences(text):
    sentences = []
    blocks = text.split("============================")

    for block in blocks:
        sents = block.split("\n\n")
        for sent in sents:
            sent = sent.replace("\n", "").replace("[", "").replace("]", "")
            if sent != "":
                sentences.append(sent)

    return sentences


def load_treebank_splits(datadir):
    train = []
    dev = []
    test = []

    print("Loading treebank data from:", datadir)
    if not os.path.isdir(datadir):
        raise FileNotFoundError(f"datadir does not exist: {datadir}")

    for subdir, dirs, files in os.walk(datadir):
        for filename in files:
            if filename.endswith(".pos"):
                filepath = os.path.join(subdir, filename)
                # Debug: show that we actually find files
                # print("Found .pos file:", filepath)
                with open(filepath, "r") as fh:
                    text = fh.read()
                    section = int(os.path.basename(subdir))

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
        self.alpha = alpha
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
                vocab.add(word)
                tagset.add(tag)

        vocab.add(self.unk_token)

        self.tags = sorted(tagset)
        self.vocab = sorted(vocab)

        trans_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        emit_counts = collections.defaultdict(lambda: collections.defaultdict(int))

        for sent in train_sents:
            prev_tag = self.start_tag
            for word, tag in sent:
                trans_counts[prev_tag][tag] += 1
                emit_counts[tag][word] += 1
                prev_tag = tag

            trans_counts[prev_tag][self.stop_tag] += 1

        self.log_trans = {}
        self.log_emit = {}

        possible_next = self.tags + [self.stop_tag]
        num_next = len(possible_next)
        prev_tags_for_trans = [self.start_tag] + self.tags

        for prev_tag in prev_tags_for_trans:
            self.log_trans[prev_tag] = {}
            total = 0
            for next_tag in possible_next:
                total += trans_counts[prev_tag][next_tag]

            denom = total + self.alpha * num_next

            for next_tag in possible_next:
                count = trans_counts[prev_tag][next_tag]
                prob = (count + self.alpha) / denom
                self.log_trans[prev_tag][next_tag] = math.log(prob)

        vocab_size = len(self.vocab)

        for tag in self.tags:
            self.log_emit[tag] = {}
            total = 0
            for word in emit_counts[tag]:
                total += emit_counts[tag][word]

            denom = total + self.alpha * vocab_size

            for word in self.vocab:
                count = emit_counts[tag][word]
                prob = (count + self.alpha) / denom
                self.log_emit[tag][word] = math.log(prob)

    def _viterbi(self, words):
        norm_words = [w if w in self.vocab else self.unk_token for w in words]
        n = len(norm_words)

        if n == 0:
            return []

        dp = [dict() for _ in range(n)]
        backptr = [dict() for _ in range(n)]

        w0 = norm_words[0]

        for tag in self.tags:
            trans = self.log_trans[self.start_tag][tag]
            emit = self.log_emit[tag][w0]
            dp[0][tag] = trans + emit
            backptr[0][tag] = self.start_tag

        for j in range(1, n):
            wj = norm_words[j]
            for curr_tag in self.tags:
                best_score = float("-inf")
                best_prev = None
                for prev_tag in self.tags:
                    score = dp[j - 1][prev_tag] + self.log_trans[prev_tag][curr_tag] + self.log_emit[curr_tag][wj]
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag

                dp[j][curr_tag] = best_score
                backptr[j][curr_tag] = best_prev

        best_final_score = float("-inf")
        best_last_tag = None

        for prev_tag in self.tags:
            score = dp[n - 1][prev_tag] + self.log_trans[prev_tag][self.stop_tag]
            if score > best_final_score:
                best_final_score = score
                best_last_tag = prev_tag

        best_tags = [None] * n
        best_tags[-1] = best_last_tag

        for j in range(n - 1, 0, -1):
            best_tags[j - 1] = backptr[j][best_tags[j]]

        return best_tags

    def tag_sentence(self, words):
        tags = self._viterbi(words)
        return list(zip(words, tags))


def main():
    datadir = os.path.join("data", "penn-treebank3-wsj", "wsj")

    train, dev, test = load_treebank_splits(datadir)

    train_sents = [get_token_tag_tuples(sent) for sent in train]
    dev_sents = [get_token_tag_tuples(sent) for sent in dev]
    test_sents = [get_token_tag_tuples(sent) for sent in test]

    hmm = HMMTagger(alpha=1.0)
    print("Training HMM tagger.")
    hmm.fit(train_sents)

    print("Tagging test set with HMM.")
    tagged_test_sentences = [
        hmm.tag_sentence([token for token, tag in sentence])
        for sentence in test_sents
    ]

    print("Evaluation (HMM vs gold):")
    evaluate(test_sents, tagged_test_sentences)


if __name__ == "__main__":
    main()
