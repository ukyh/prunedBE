# pruned Basic Elements (pBE)

The code of pBE, an automatic evaluation measure of summarization. Details are described in the following paper:

[**Pruning Basic Elements for Better Automatic Evaluation of Summaries**](http://aclweb.org/anthology/N18-2104)  
Ukyo Honda, Tsutomu Hirao and Masaaki Nagata  
In *Proceedings of NAACL*, 2018


## Requirements

* python (tested with 3.5.3)
* gensim (tested with 3.5.0)
* scikit-learn (tested with 0.18.1)
* java (tested with 1.8.0_144)
* stanford-corenlp (tested with 3.8.0)


## Input Structure

The reference and target summaries have to be organized as follows:
```
root/
 └── dataset/
      ├── ref/
      │    ├── reference summary 1
      │    ├── reference summary 2
      │    └── ...
      ├── trg/
      │    ├── target summary 1
      │    ├── target summary 2
      │    └── ...
      ├── ref_parsed/
      ├── trg_parsed/
      ├── cluster/
      └── score/
```
**NOTE:**  
We assume the file names of the summaries to be like those of the DUC and TAC summaries. That is, the file names of the summaries ***start with topic id***, end with reference/system id, and these ids are ***connected by period*** (e.g., `D30003.M.100.T.B`).


## Preprocess

### Parsing

Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html#download).

Parse the summaries.
```
# compile
javac -cp ".:PATH_TO_CORENLP" Parser.java

# run
java -cp ".:PATH_TO_CORENLP" Parser DATASET_NAME
```
**NOTE:**  
`PATH_TO_CORENLP` is a path to the whole items in CoreNLP (e.g., `./stanford-corenlp-full-2017-06-09/*`).

### Clustering

Download the word embeddings pre-trained on GoogleNews ([GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)) and install [gensim](https://radimrehurek.com/gensim/) and [scikit-learn](https://scikit-learn.org/stable/install.html).

Apply clustering on the parsed summaries.
```
python -u clustering.py \
    --dataset DATASET_NAME \
    --ref_dir ref_parsed \
    --trg_dir trg_parsed \
    --cls_dir cluster \
    --rel_path ./relations.txt \
    --cluster_rate 0.975 \
    --affinity cosine \
    --linkage complete \
    --emb_path ./GoogleNews-vectors-negative300.bin
```


## Run

Run pBE. Command below corresponds to `pBE_{-cnt+cls}` described in the paper.
```
python -u pBE.py \
    --dataset DATASET_NAME \
    --ref_dir ref_parsed \
    --trg_dir trg_parsed \
    --cls_dir cluster \
    --out_dir score \
    --rel_path ./relations.txt \
    --ignore_freq \
    --assign_cluster
```


## Citation
```
@inproceedings{honda2018pruning,
  title={Pruning Basic Elements for Better Automatic Evaluation of Summaries},
  author={Honda, Ukyo and Hirao, Tsutomu and Nagata, Masaaki},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  pages={661--666},
  year={2018}
}
```
