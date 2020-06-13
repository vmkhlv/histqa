from .utils import tokenizer, singleton
from pymorphy2 import MorphAnalyzer


START_TOKEN = "B-MISC"
INSIDE_TOKEN = "I-MISC"
OUTSIDE_TOKEN = "O"
FILLER = "@"


@singleton
class Morph:
    def __init__(self, from_entity_list):
        self.vocab = self.build_vocab(from_entity_list)
        self.max_len = max([len(ent) for ent in self.vocab])
        self.tokenizer = tokenizer
        self.m = MorphAnalyzer()

    def build_vocab(self, from_entity_list):
        return [
            [
                self.m.parse(e)[0].normal_form for e in self.tokenizer.tokenize(ent.lower())
            ] for ent in from_entity_list
        ]

    def index_annotate(self, text):
        tokenized = [FILLER] * self.max_len + self.tokenizer.tokenize(text.lower())
        tags = [OUTSIDE_TOKEN] * len(tokenized)
        for i, token in enumerate(tokenized):
            lemma = self.m.parse(token)[0].normal_form
            # get all entity candidates containing the token
            if any([lemma in entity for entity in self.vocab]):
                candidates = [entity for entity in self.vocab if lemma in entity]
                # greedy annotation starting from the longest entity
                candidates.sort(key=lambda entity: len(entity), reverse=True)
                for candidate in candidates:
                    # early annotation if there is only one candidate
                    if len(candidates) == 1 and " ".join(candidates[0]) == lemma:
                        tags[i] = INSIDE_TOKEN
                    else:
                        # checking if the candidate is within the window
                        window_size = len(candidate)
                        left_ngram = max(i - window_size + 1, 0)
                        right_ngram = min(i + window_size, len(tokenized) - 1)
                        window = tokenized[left_ngram:right_ngram]
                        window_lemmas = [self.m.parse(token)[0].normal_form for token in window]
                        candidate_len = len(candidate)
                        for j in range(len(window_lemmas)):
                            curr_stride = window_lemmas[j:j + candidate_len]
                            # reduce stride space
                            if len(curr_stride) >= candidate_len and curr_stride == candidate:
                                if candidate_len > 2:
                                    if tags[i] == OUTSIDE_TOKEN:
                                        tags[i] = START_TOKEN
                                    for inner_position in range(i + 1, i + j):
                                        if tags[inner_position] == OUTSIDE_TOKEN:
                                            tags[inner_position] = INSIDE_TOKEN
                                    tags[i + j] = INSIDE_TOKEN
                                elif candidate_len == 2:
                                    tags[i] = START_TOKEN
                                    tags[i + j] = INSIDE_TOKEN
                                else:
                                    tags[i] = INSIDE_TOKEN
        return {"text": " ".join([t for t in tokenized if t != FILLER]),
                "tags": tags[self.max_len:]}
