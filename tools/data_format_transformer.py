import os
from collections import namedtuple, Counter

import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from sentence_splitter import SentenceSplitter


sentence_splitter = SentenceSplitter(language='es')


class Word:
    def __init__(self, word_text, word_begin, word_end):
        self.word_text = word_text
        self.word_begin = word_begin
        self.word_end = word_end
        self.label = 'O'


LabelsObject = namedtuple('LabelsObject', ['label_text', 'label_begin', 'label_end', 'src_str'])


def get_markup(file_with_markup):
    with open(file_with_markup, 'r') as f:
        data = f.readlines()
        data = [l.strip().split('\t')[1:] for l in data]
        data = [
            LabelsObject(
                el[0].split(' ')[0],
                int(el[0].split(' ')[1:][0]),
                int(el[0].split(' ')[1:][1]),
                el[-1]) for el in data
        ]
        return data


def get_text(file_with_text):
    with open(file_with_text, 'r') as f:
        data = f.readlines()
        return '$'.join([el.replace('\n', '') for el in data])


def markup_aligner(markup, text):
    sent_with_markup = []
    sentences = text.split('$')
    words_borders = list(WordPunctTokenizer().span_tokenize(text))
    for sentence in sentences:
        tokenized_sentence = []
        sentence_begin = text.find(sentence)
        sentence_end = sentence_begin + len(sentence)
        for word_begin, word_end in words_borders:
            if word_begin >= sentence_begin and word_end <= sentence_end:
                word_text = text[word_begin: word_end]
                word = Word(word_text, word_begin, word_end)
                for label in markup:
                    if word.word_begin >= label.label_begin and word.word_end <= label.label_end:
                        word.label = label.label_text
                tokenized_sentence.append(word)
        sent_with_markup.append(tokenized_sentence)
    return sent_with_markup


init_data_path = '/home/m.domrachev/repos/competitions/MEDDOCAN/data//'
result_data_path = '/home/m.domrachev/repos/competitions/MEDDOCAN/data/markups//'

files = []
for f in os.listdir(init_data_path):
    files.append([os.path.join(init_data_path, f), f.split('.')[0]])

group_mark_files = dict()
for _f in files:
    group_mark_files[_f[-1]] = group_mark_files.setdefault(_f[-1], []) + [_f[0]]

labels = []
for fg in group_mark_files:

    # if fg == 'S0004-06142005000900013-1':

        markup_data = sorted(group_mark_files[fg])
        markup = get_markup(markup_data[0])
        text = get_text(markup_data[1])
        sent_with_markup = markup_aligner(markup, text)

        with open(os.path.join(result_data_path, fg + '.txt'), 'w+') as rf:
            for s in sent_with_markup:
                for t in s:
                    labels.append(t.label)
                    rf.write(t.word_text + ' ' + t.label + '\n')
                rf.write('\n')

labels = Counter(labels)
data_freq = pd.DataFrame()
data_freq['label'] = [el for el in labels]
data_freq['#'] = [labels[el] for el in labels]
print(data_freq.sort_values(by=['#']))
