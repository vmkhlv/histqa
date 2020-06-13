# Master Thesis: Exploring Language Patterns for Multi-hop Question Answering

#### Hybrid Approach: Multi-hop/One-hop Question Answering over a Knowledge Graph and an Entity-linked Document Collection

##### Vladislav Mikhailov, Higher School of Economics, Computational Linguistics, 2020

The repository contains the main code and datasets used in the experiments. Link to pytorch [NER model](https://yadi.sk/d/V5k7H1y1xN8ogg).

The repository is structured as follows.
* data:
    - question_subgraphs: one-hop inferences structured as a graph;
    - tasks: tasks from the Unified State History Exam with labeled task choices;
    - multi_hop_questions.json: natural language multi-hop questions labeled with "S" tag for BERT-based decomposition;
    - one_hop_questions.json: sub-questions from the Unified State History Exam tasks (labeled sub-question & answer pairs);

* models:
    - elmo.py: the embedder for DeepPavlov ELMo trained on Wikipedia;
    - histqanet.py: single-hop QA model;
    - morph.py: morphology-based algorithm for automatic NER annotation;
    - ner.py: domain-specific NER model;
    - utils.py: supplementary \& preprocessing functions;
