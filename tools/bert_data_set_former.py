import os
import itertools

import pandas as pd


class LemmaDataSetCreator:
    def __init__(self):
        self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.markup_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/bert_markup/'))
        self.train = self.sent_pair2sentence_tokens(os.path.join(self.project_path + '/data/init_data/train/brat/'))
        self.dev = self.sent_pair2sentence_tokens(os.path.join(self.project_path + '/data/init_data/dev/brat/'))
        self.sent_pair2sentence_tokens_test(os.path.join(self.project_path + '/data/init_data/test/'))
        self.save_data_set()

    def sent_pair2sentence_tokens(self, path2data):
        sent = []
        for file in os.listdir(path2data):
            if file.endswith('.conll'):
                with open(os.path.join(path2data + file), "r") as file_object:
                    data = [list(y) for x, y in
                            itertools.groupby([line for line in file_object.readlines()], lambda z: z == '\n')]
                    data = [[t.replace('\n', '') for t in s] for s in data]
                    data = [[t.split('\t') for t in s if len(t.split()) > 1] for s in data if s != ['']]

                    for seq in data:
                        tokens = [el[0].replace(' ', '_') for el in seq]
                        labels = [self.preprocessed_label(el[-1]) for el in seq]   #
                        sent.append((' '.join(labels), ' '.join(tokens)))
        return sent

    def sent_pair2sentence_tokens_test(self, path2data):
        for file in os.listdir(path2data):
            sent = []
            if file.endswith('.conll'):
                with open(os.path.join(path2data + file), "r") as file_object:
                    data = [list(y) for x, y in
                            itertools.groupby([line for line in file_object.readlines()], lambda z: z == '\n')]
                    data = [[t.replace('\n', '') for t in s] for s in data]
                    data = [[t.split('\t') for t in s if len(t.split()) > 1] for s in data if s != ['']]

                    for seq in data:
                        tokens = [el[0].replace(' ', '_') for el in seq]
                        labels = [self.preprocessed_label(el[-1]) for el in seq]   #
                        sent.append((' '.join(labels), ' '.join(tokens)))
            test_df = pd.DataFrame(self.dev, columns=["0", "1"])
            test_df.to_csv(self.markup_path + "/test/%s.csv" % (file.split('.')[0],), index=False)

    def preprocessed_label(self, label):
        if label == 'O':
            return label
        else:
            bio_tag = label.split('-')[0]
            _class = label.split('-')[-1].replace('_', '-')
            return bio_tag + '_' + _class

    def save_data_set(self):
        train_df = pd.DataFrame(self.train, columns=["0", "1"])
        train_df.to_csv(self.markup_path + "/train.csv", index=False)
        valid_df = pd.DataFrame(self.dev, columns=["0", "1"])
        valid_df.to_csv(self.markup_path + "/valid.csv", index=False)


if __name__ == '__main__':
    LemmaDataSetCreator()
