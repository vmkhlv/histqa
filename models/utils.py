import numpy as np
from spacy.lang.ru import Russian
from tqdm.auto import tqdm
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
import pickle
import re
from .elmo import ElmoEmbedder as elmo_emb
from functools import wraps


def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return inner


def clean_text(text):
    text = text.replace(u"\xa0", " ").replace("†", " ").replace("•", " ")
    text = text.translate(
        str.maketrans({key: " {0} ".format(key) for key in '"-!&()*+,/:;<=>?[]^`{|}~'})
    )
    rep_keys = {
        "и́": "и",
        "а́": "а",
        "е́": "е",
        "ю́": "ю",
        "у́": "у",
        "ы́": "ы",
        "о́": "о",
        "я́": "я",
    }
    for rep in rep_keys:
        text = text.replace(rep, rep_keys[rep])
    text = re.sub("\d{2,}", " ", text).replace("\n", "")
    return text


@singleton
class SpacyTokenizer:
    def __init__(self):
        self.nlp = Russian()
        self.nlp.add_pipe(
            RussianTokenizer(self.nlp, MERGE_PATTERNS), name="russian_tokenizer"
        )

    def tokenize(self, text):
        return [token.text for token in self.nlp(text) if token.text.strip()]


tokenizer = SpacyTokenizer()


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def create_batches(data, n):
    return [data[x : x + n] for x in range(0, len(data), n)]


def check(doc, w_i, cand):
    return sum(
        [doc[w_i + j].lower() == c_ for j, c_ in enumerate(cand) if w_i + j < len(doc)]
    ) == len(cand)


def ind(s_i, w_i, c_i, cand):
    return [[s_i, w_i + i, c_i] for i in range(len(cand))]


def prepare_data(data, tokenizer=tokenizer):
    res = []
    for sample in tqdm(data):
        try:
            sample["query"] = tokenizer.tokenize(sample["query"])
            sample["candidates_origin"] = list(sample["candidates"])
            sample["candidates"] = [
                tokenizer.tokenize(cand) for cand in sample["candidates"]
            ]
            sample["documents"] = [
                tokenizer.tokenize(clean_text((doc))) for doc in sample["documents"]
            ]
            mask = [
                [
                    ind(s_i, w_i, c_i, cand)
                    for w_i, w in enumerate(doc)
                    for c_i, cand in enumerate(sample["candidates"])
                    if check(doc, w_i, cand)
                ]
                for s_i, doc in enumerate(sample["documents"])
            ]

            node_ids = []
            cand = 0
            for e in [[[x[-1] for x in cand][0] for cand in doc] for doc in mask]:
                u = []
                for f in e:
                    u.append((cand, f))
                    cand += 1
                node_ids.append(u)

            sample["nodes_candidates_id"] = [
                [x[-1] for x in f][0] for e in mask for f in e
            ]
            sample["answer_candidates_id"] = [
                i
                for i in sample["nodes_candidates_id"]
                if sample["cand_origin"][i] == sample["answer"]
            ]

            if not len(set(sample["answer_candidates_id"])) == 1:
                continue

            sample["answer_candidates_id"] = sample["answer_candidates_id"][0]

            edges_in, edges_out = [], []

            for e0 in node_ids:
                for f0, w0 in e0:
                    for f1, w1 in e0:
                        if f0 != f1:
                            edges_in.append((f0, f1))

                    for e1 in node_ids:
                        for f1, w1 in e1:
                            if e0 != e1 and w0 == w1:
                                edges_out.append((f0, f1))

            sample["edges_in"] = edges_in
            sample["edges_out"] = edges_out

            masked_nodes = [[x[:-1] for x in f] for e in mask for f in e]
            candidates = elmo_emb.batch_to_embeddings(sample["documents"])
            sample["nodes_elmo"] = [
                (candidates[np.array(m).T.tolist()]).astype(np.float16)
                for m in masked_nodes
            ]
            query = elmo_emb.batch_to_embeddings([sample["query"]])
            sample["query_elmo"] = (query).astype(np.float16)[0]
            res.append(sample)
        except Exception as e:
            print(e)
    return res
