import os
import re
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str, required=True, help="Dataset name")
    parser.add_argument('--ref_dir', default='ref_parsed', type=str, help="Directory of the parsed reference summaries")
    parser.add_argument('--trg_dir', default='trg_parsed', type=str, help="Directory of the parsed target summaries")
    parser.add_argument('--cls_dir', default='cluster', type=str, help="Directory of the clustering output")
    parser.add_argument('--out_dir', default='score', type=str, help="Directory of the pBE output")
    parser.add_argument('--rel_path', default='./relations.txt', type=str, help="Path to the selected UD relations")
    parser.add_argument('--ignore_freq', default=False, action='store_true', help="`-cnt` option described in the paper")
    parser.add_argument('--assign_cluster', default=False, action='store_true', help="`+cls` option described in the paper")
    args = parser.parse_args()
    return args


class BE():

    def __init__(self, args):
        self.ref_path = os.path.join(args.dataset, args.ref_dir)
        self.trg_path = os.path.join(args.dataset, args.trg_dir)
        self.cls_path = os.path.join(args.dataset, args.cls_dir)
        self.out_path = os.path.join(args.dataset, args.out_dir)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.ref_fname_dict = self.get_ref()
        self.trg_fname_dict = self.get_trg()
        self.ref_count = dict()
        self.trg_count = dict()
        self.cls_dict = dict()

        with open(args.rel_path) as f:
            self.rel_set = [i.rstrip() for i in f]
        self.position = re.compile('-[0-9]+$')
        self.extention = re.compile('.txt|.html')

        self.ignore_freq = args.ignore_freq
        self.assign_cluster = args.assign_cluster

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
        for topic, ref_fnames in self.ref_fname_dict.items():
            if topic in self.trg_fname_dict:
                trg_fnames = self.trg_fname_dict[topic]
                yield topic, ref_fnames, trg_fnames
            else:
                continue

    # get cluster id dict for each topic
    def get_cls_dict(self, trg_fname):
        self.cls_dict = dict()
        if self.assign_cluster:
            with open(os.path.join(self.cls_path, trg_fname)) as cls_file:
                for line in cls_file:
                    try:
                        word, ID = line.rstrip().split('\t')
                        self.cls_dict[word] = ID
                    except:
                        word = line.rstrip()
                        self.cls_dict[word] = word

    # get BE count on a reference set
    def get_ref_count(self, ref_fnames):
        self.ref_count = dict()
        counter = 0
        for ref_name in ref_fnames:
            with open(os.path.join(self.ref_path, ref_name)) as ref:
                for line in ref:
                    if len(line.rstrip()) > 0:
                        rel, tokens = line.rstrip().split('(')
                        try:
                            rel_head = rel.split(':')[0]
                        except:
                            rel_head = rel
                        if rel_head in self.rel_set:
                            tmp1,tmp2 = tokens.rstrip(')').split(', ')
                            token1 = self.position.sub('', tmp1)
                            token2 = self.position.sub('', tmp2)
                            if self.assign_cluster: # convert to cluster idx
                                token1 = self.cls_dict[token1]
                                token2 = self.cls_dict[token2]
                            be = '{} {} {}'.format(rel, token1, token2)
                            if be in self.ref_count:
                                self.ref_count[be][counter] += 1  # one reference one column
                            else:
                                self.ref_count[be] = [0 for _ in range(len(ref_fnames))]
                                self.ref_count[be][counter] = 1
            counter += 1    # next reference (next column)

    # get BE count on a target
    def get_trg_count(self, trg_fname):
        self.trg_count = dict()
        with open(os.path.join(self.trg_path, trg_fname)) as trg:
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
                        if self.assign_cluster:
                            token1 = self.position.sub('', tmp1)
                            token2 = self.position.sub('', tmp2)
                            if self.assign_cluster: # convert to cluster idx
                                token1 = self.cls_dict[token1]
                                token2 = self.cls_dict[token2]
                        be = '{} {} {}'.format(rel, token1, token2)
                        if be in self.trg_count:  # not first encount
                            self.trg_count[be] += 1
                        else:                     # first encount
                            if be in self.ref_count:
                                self.trg_count[be] = 1

    # get BE denominator
    def get_denominator(self):
        denominator = 0
        for unit, freq in self.ref_count.items():
            if not self.ignore_freq:
                denominator += sum(freq)
            else:
                denominator += sum([1 for i in freq if i > 0])
        return denominator

    # get BE numerator
    def get_numerator(self):
        numerator = 0
        for unit in self.trg_count:
            if unit in self.ref_count:
                trg_freq = self.trg_count[unit]
                if self.ignore_freq:
                    trg_freq = 1
                for ref_freq in self.ref_count[unit]:
                    numerator += min(trg_freq, ref_freq)
        return numerator

    # output BE score
    def output_score(self, out_file):
        denominator = self.get_denominator()
        numerator = self.get_numerator()
        score = numerator / denominator
        fname = self.extention.sub('', trg_fname)   # for some dataset such as DUC03
        out_file.write('{}\t{}\n'.format(fname, score))




if __name__ == '__main__':
    args = get_args()
    measure = BE(args)
    for topic, ref_fnames, trg_fnames in measure.gen_pair():
        with open(os.path.join(measure.out_path, topic + '.txt'), 'w') as out_file:
            for trg_fname in trg_fnames:
                if trg_fname not in ref_fnames: # exclude reference files from target
                    measure.get_cls_dict(trg_fname)
                    measure.get_ref_count(ref_fnames)
                    measure.get_trg_count(trg_fname)
                    measure.output_score(out_file)
