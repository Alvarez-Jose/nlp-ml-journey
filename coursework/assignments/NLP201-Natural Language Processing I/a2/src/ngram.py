import math
from collections import Counter, defaultdict
from data import load_split, SimpleTokenizer

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"


class NgramLM:
    def __init__(self):
        self.uni_counts = Counter()
        self.bi_counts = Counter()
        self.tri_counts = Counter()

        self.bi_context_counts = Counter()
        self.tri_context_counts = Counter()

        self.total_unigrams = 0

    def train(self, sentences):
        """
        sentences: list of tokenized sentences with
        [<START>, <START>, ..., <STOP>]
        """
        for sent in sentences:
            for i in range(2, len(sent)):
                w_im2, w_im1, w_i = sent[i-2], sent[i-1], sent[i]

                # unigram ignores STARTs
                if w_i != START:
                    self.uni_counts[w_i] += 1
                    self.total_unigrams += 1

                # bigram
                self.bi_counts[(w_im1, w_i)] += 1
                self.bi_context_counts[w_im1] += 1

                # trigram
                self.tri_counts[(w_im2, w_im1, w_i)] += 1
                self.tri_context_counts[(w_im2, w_im1)] += 1

    def p_unigram(self, w):
        c = self.uni_counts[w]
        if self.total_unigrams == 0:
            return 0.0
        return c / self.total_unigrams

    def p_bigram(self, w_prev, w):
        num = self.bi_counts[(w_prev, w)]
        denom = self.bi_context_counts[w_prev]
        if denom == 0:
            return 0.0
        return num / denom

    def p_trigram(self, w_im2, w_im1, w_i):
        num = self.tri_counts[(w_im2, w_im1, w_i)]
        denom = self.tri_context_counts[(w_im2, w_im1)]
        if denom == 0:
            return 0.0
        return num / denom


class InterpolatedLM:
    def __init__(self, base_lm, lambda1: float, lambda2: float, lambda3: float):
        assert abs((lambda1 + lambda2 + lambda3) - 1.0) < 1e-8
        self.lm = base_lm
        self.l1 = lambda1
        self.l2 = lambda2
        self.l3 = lambda3

    def p(self, w_im2, w_im1, w_i):
        # unigram component
        p1 = self.lm.p_unigram(w_i)

        # bigram component
        p2 = self.lm.p_bigram(w_im1, w_i)

        # trigram component *handles the first non start token
        if w_im2 == START and w_im1 == START:
            # the trigram model uses bigram p(w | <START>)
            p3 = self.lm.p_bigram(START, w_i)
        else:
            p3 = self.lm.p_trigram(w_im2, w_im1, w_i)

        return self.l1 * p1 + self.l2 * p2 + self.l3 * p3


def sentence_log_prob_unigram(model, sent):
    logprob = 0.0
    M = 0
    for w in sent:
        if w == START:
            continue
        p = model.p_unigram(w)
        if p == 0:
            # tiny floor just so log()
            p = 1e-12
        logprob += math.log(p)
        M += 1
    return logprob, M


def sentence_log_prob_bigram(model, sent):
    logprob = 0.0
    M = 0
    for i in range(1, len(sent)):
        w_prev, w = sent[i-1], sent[i]
        if w == START:
            continue
        p = model.p_bigram(w_prev, w)
        if p == 0:
            p = 1e-12
        logprob += math.log(p)
        M += 1
    return logprob, M


def sentence_log_prob_trigram(model, sent):
    logprob = 0.0
    M = 0
    for i in range(2, len(sent)):
        w_im2, w_im1, w_i = sent[i-2], sent[i-1], sent[i]
        if w_i == START:
            continue
        # token immediately after <START>, use bigram p(w | <START>)
        if w_im2 == START and w_im1 == START:
            p = model.p_bigram(START, w_i)
        else:
            p = model.p_trigram(w_im2, w_im1, w_i)

        if p == 0:
            p = 1e-12
        logprob += math.log(p)
        M += 1
    return logprob, M


def perplexity(model, sentences, order=1):
    total_logprob = 0.0
    total_M = 0
    for sent in sentences:
        if order == 1:
            lp, M = sentence_log_prob_unigram(model, sent)
        elif order == 2:
            lp, M = sentence_log_prob_bigram(model, sent)
        elif order == 3:
            lp, M = sentence_log_prob_trigram(model, sent)
        else:
            raise ValueError("order must be 1, 2, or 3")

        total_logprob += lp
        total_M += M

    avg_logprob = total_logprob / total_M
    return math.exp(-avg_logprob)


def perplexity_interpolated(interp_lm, sentences):
    total_logprob = 0.0
    total_M = 0
    for sent in sentences:
        for i in range(2, len(sent)):
            w_im2, w_im1, w_i = sent[i-2], sent[i-1], sent[i]
            if w_i == START:
                continue
            p = interp_lm.p(w_im2, w_im1, w_i)
            if p == 0:
                p = 1e-12
            total_logprob += math.log(p)
            total_M += 1

    avg_logprob = total_logprob / total_M
    return math.exp(-avg_logprob)


def hdtv_sanity_check(model):
    sent = [START, START, "HDTV", ".", STOP]
    sents = [sent]
    uni_ppl = perplexity(model, sents, order=1)
    bi_ppl  = perplexity(model, sents, order=2)
    tri_ppl = perplexity(model, sents, order=3)
    print("HDTV sanity ppl -> uni: {:.1f}, bi: {:.1f}, tri: {:.1f}".format(
        uni_ppl, bi_ppl, tri_ppl
    ))


def main():
    # Load raw splits
    train_raw = load_split("train")
    dev_raw   = load_split("dev")
    test_raw  = load_split("test")

    # Build vocab and preprocess 
    tokenizer = SimpleTokenizer(min_freq=3)
    tokenizer.build_vocabulary(train_raw)

    assert tokenizer.vocab is not None
    print("Vocab size (including <UNK> and <STOP>, excluding <START>):",
          len(tokenizer.vocab))

    train_proc = tokenizer.preprocess_train(train_raw)
    dev_proc   = tokenizer.preprocess_other(dev_raw)
    test_proc  = tokenizer.preprocess_other(test_raw)

    # Train LM
    lm = NgramLM()
    lm.train(train_proc)

    # Perplexities on train test for unigram, bigram, trigram
    for order, name in [(1, "Unigram"), (2, "Bigram"), (3, "Trigram")]:
        train_ppl = perplexity(lm, train_proc, order=order)
        dev_ppl   = perplexity(lm, dev_proc, order=order)
        test_ppl  = perplexity(lm, test_proc, order=order)

        print(f"{name} PPL -> train: {train_ppl:.2f}, "
              f"dev: {dev_ppl:.2f}, test: {test_ppl:.2f}")

    # Linear interpolation experiments

    lambda_sets = [
        (0.3, 0.3, 0.4),
        (0.1, 0.3, 0.6),
        (0.2, 0.3, 0.5),
        (0.1, 0.2, 0.7),
        (0.05, 0.25, 0.70),
    ]

    print("\nInterpolated trigram perplexities (train/dev):")
    best_dev = float("inf")
    best_lambdas = None

    for l1, l2, l3 in lambda_sets:
        interp = InterpolatedLM(lm, l1, l2, l3)
        train_ppl_interp = perplexity_interpolated(interp, train_proc)
        dev_ppl_interp   = perplexity_interpolated(interp, dev_proc)

        print(f"λ1={l1:.2f}, λ2={l2:.2f}, λ3={l3:.2f} -> "
              f"train {train_ppl_interp:.2f}, dev {dev_ppl_interp:.2f}")

        if dev_ppl_interp < best_dev:
            best_dev = dev_ppl_interp
            best_lambdas = (l1, l2, l3)

    if best_lambdas is not None:
        l1, l2, l3 = best_lambdas
        best_interp = InterpolatedLM(lm, l1, l2, l3)
        test_ppl_interp = perplexity_interpolated(best_interp, test_proc)
        print(f"\nBest λ on dev: λ1={l1:.2f}, λ2={l2:.2f}, λ3={l3:.2f}")
        print(f"Interpolated trigram test perplexity: {test_ppl_interp:.2f}")


if __name__ == "__main__":
    main()
