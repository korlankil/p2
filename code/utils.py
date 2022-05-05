from torch.utils.data import IterableDataset, DataLoader
import logging
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from models import Bert
logging.basicConfig(filename='../log/logging.txt', level=logging.INFO)


class Dataset(IterableDataset):
    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):
        super(Dataset, self).__init__()

        bert=Bert(BertModel, 'bert-base-uncased', opt.bert_path )

        logging.info("Reading data from %s" % src_file)
        with open(src_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        self.data = []
        for ind, doc in enumerate(tqdm(raw_data)):
            title, entity_list, labels, sentences = doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents']

            # generate positive examples
            train_triple = []
            new_labels = []
            for label in labels:
                head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence']
                assert relation in rel2id, "no such relation {} in rel2id".format(relation)
                train_triple.append((head, tail))
                # relation mapping
                label['r'] = rel2id[relation]
                label['in_train'] = False

                # record training set mention triples and mark it for dev and test set
                for n1 in entity_list[head]:
                    for n2 in entity_list[tail]:
                        mention_triple = (n1['name'], n2['name'], relation)
                        if dataset_type == 'train':
                            self.instance_in_train.add(mention_triple)
                        else:
                            if mention_triple in self.instance_in_train:
                                label['in_train'] = True
                                break

                new_labels.append(label)

            # generate negative examples
            nagetive_triple = []
            for j in range(len(entity_list)):
                for k in range(len(entity_list)):
                    if j != k and (j, k) not in train_triple:
                        nagetive_triple.append((j, k))

            # generate document ids
            words = []
            for sentence in sentences:
                for word in sentence:
                    words.append(word)
            if len(words) > self.document_max_length:
                words = words[:self.document_max_length]

            word_id = np.zeros((self.document_max_length,), dtype=np.int32)
            pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
            ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
            mention_id = np.zeros((self.document_max_length,), dtype=np.int32)

            # word to id
            for w_ind, w in words:
                word = word2id.get(w.lower(), word2id['UNK'])
                word_id[w_ind] = word

            # entity to mention sentence
            entity2mention = defaultdict(list)
            mention_idx = 1
            already_exist = set()
            for idx, vertext in enumerate(entity_list,1):
                for v in vertext:
                    sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']
                    pos0 = bert_starts[pos0]
                    pos1 = bert_starts[pos1]

