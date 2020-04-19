import os
import re
import logging
from argparse import ArgumentParser
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import AgglomerativeClustering


logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str, required=True, help="Dataset name")
    parser.add_argument('--ref_dir', default='ref_parsed', type=str, help="Directory of the parsed reference summaries")
    parser.add_argument('--trg_dir', default='trg_parsed', type=str, help="Directory of the parsed target summaries")
    parser.add_argument('--cls_dir', default='cluster', type=str, help="Directory of the clustering output")
    parser.add_argument('--rel_path', default='./relations.txt', type=str, help="Path to the selected UD relations")
    parser.add_argument(
        '--cluster_rate', default=0.975, type=float,
        help="Rate to determine cluster numbers (`cluster_num = cluster_rate * unigram_num`)"
    )
    parser.add_argument(
        '--affinity', default='cosine', type=str,
        help="Metric to compute linkage between word embeddings (See `affinity` hyperparameter of `sklearn.cluster.AgglomerativeClustering`)"
    )
    parser.add_argument(
        '--linkage', default='complete', type=str,
        help="Criterion on linkage between clusters (See `linkage` hyperparameter of `sklearn.cluster.AgglomerativeClustering`)"
    )
    parser.add_argument('--emb_path', default='./GoogleNews-vectors-negative300.bin', help="Path to word embeddings")
    args = parser.parse_args()
    return args


class ClusterModel():

    def __init__(self, args):
        self.ref_path = os.path.join(args.dataset, args.ref_dir)
        self.trg_path = os.path.join(args.dataset, args.trg_dir)
        self.cls_path = os.path.join(args.dataset, args.cls_dir)
        if not os.path.exists(self.cls_path):
            os.makedirs(self.cls_path)

        self.ref_fname_dict = self.get_ref()
        self.trg_fname_dict = self.get_trg()

        with open(args.rel_path) as f:
            self.rel_set = [i.rstrip() for i in f]
        self.position = re.compile('-[0-9]+$')
        self.extention = re.compile('.txt|.html')

        self.cluster_rate = args.cluster_rate
        self.affinity = args.affinity
        self.linkage = args.linkage
        self.embeddings = KeyedVectors.load_word2vec_format(args.emb_path, binary=True)

    # get reference file names grouped by topic id
    def get_ref(self):
        ref_files = [path for path in os.listdir(self.ref_path) if not path.startswith('.')]
        ref_fname_dict = dict()
        for file_name in ref_files:
            ref_topic = file_name.split('.')[0]
            if ref_topic in ref_fname_dict:
                ref_fname_dict[ref_topic].append(file_name)
            else:
                ref_fname_dict[ref_topic] = [file_name]
        return ref_fname_dict

    # get target file names grouped by topic id
    def get_trg(self):
        trg_files = [path for path in os.listdir(self.trg_path) if not path.startswith('.')]
        trg_fname_dict = dict()
        for file_name in trg_files:
            trg_topic = file_name.split('.')[0]
            if trg_topic in trg_fname_dict:
                trg_fname_dict[trg_topic].append(file_name)
            else:
                trg_fname_dict[trg_topic] = [file_name]
        return trg_fname_dict

    # generate a pair of references and targets in the same topic id
    def gen_pair(self):
        for topic, ref_in_topic in self.ref_fname_dict.items():
            if topic in self.trg_fname_dict:
                trg_in_topic = self.trg_fname_dict[topic]
                yield ref_in_topic, trg_in_topic
            else:
                continue

    def assign_cluster(self, ref_in_topic, trg_in_topic):
        # reference vocab
        ref_vocab = []
        for file_name in ref_in_topic:
            with open(os.path.join(self.ref_path, file_name)) as ref:
                for line in ref:
                    if len(line.rstrip()) > 0:
                        rel, tokens = line.rstrip().split('(')
                        try:
                            rel_head = rel.split(':')[0]
                        except:
                            rel_head = rel
                        if rel_head in self.rel_set:
                            tmp1, tmp2 = tokens.rstrip(')').split(', ')
                            token1 = self.position.sub('', tmp1)
                            token2 = self.position.sub('', tmp2)
                            ref_vocab.append(token1)
                            ref_vocab.append(token2)

        # target vocab -> assign cluster
        for file_name in trg_in_topic:
            if file_name not in ref_in_topic:   # exclude reference files from target
                trg_vocab = list()
                with open(os.path.join(self.trg_path, file_name)) as trg:
                    for line in trg:
                        if len(line.rstrip()) > 0:
                            rel, tokens = line.rstrip().split('(')
                            try:
                                rel_head = rel.split(':')[0]
                            except:
                                rel_head = rel
                            if rel_head in self.rel_set:
                                tmp1, tmp2 = tokens.rstrip(')').split(', ')
                                token1 = self.position.sub('', tmp1)
                                token2 = self.position.sub('', tmp2)
                                trg_vocab.append(token1)
                                trg_vocab.append(token2)

                trg_vocab.extend(ref_vocab)
                full_vocab = set(trg_vocab)
                valid_vocab = list()
                emb_matrix = list()
                OOV = list()
                for vocab in full_vocab:
                    if vocab in self.embeddings:
                        valid_vocab.append(vocab)
                        emb_matrix.append(self.embeddings[vocab])
                    else:
                        OOV.append(vocab)

                vocab_size = len(valid_vocab)
                cluster_num = int(vocab_size * self.cluster_rate)
                model = AgglomerativeClustering(
                    n_clusters=cluster_num, affinity=self.affinity, linkage=self.linkage
                )
                cluster_ids = model.fit_predict(emb_matrix)
                out_dict = dict()
                for vocab, ID in zip(valid_vocab, cluster_ids):
                    out_dict[vocab] = ID

                with open(os.path.join(self.cls_path, file_name), 'w') as out_file:
                    for vocab, ID in sorted(out_dict.items(), key=lambda x:x[1]):
                        out_file.write('{}\t#{}#\n'.format(vocab, ID))
                    for oov in OOV:
                        out_file.write('{}\n'.format(oov))

                print('--{}'.format(file_name))
                print('vocab size: {} cluster num: {}'.format(vocab_size, cluster_num))




if __name__ == '__main__':
    args = get_args()
    model = ClusterModel(args)
    for ref_in_topic, trg_in_topic in model.gen_pair():
        model.assign_cluster(ref_in_topic, trg_in_topic)
