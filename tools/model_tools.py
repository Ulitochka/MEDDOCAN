import itertools
import string

import numpy as np


class ModelTools:

    def __init__(self):
        pass

    def preparation_data_to_score(self, yh, pr, ind2labels):
        yh = yh.argmax(2)
        pr = [list(np.argmax(el, axis=1)) for el in pr]
        coords = [np.where(yhh > 0)[0][0] for yhh in yh]
        yh = [yhh[co:] for yhh, co in zip(yh, coords)]
        ypr = [prr[co:] for prr, co in zip(pr, coords)]

        fyh_sent = [[ind2labels.get(t, 'O') for t in s] for s in yh]
        fpr_sent = [[ind2labels.get(t, 'O') for t in s] for s in ypr]

        fyh = [t for s in fyh_sent for t in s]
        fpr = [t for s in fpr_sent for t in s]

        return fyh, fpr

    def remove_null(self, true_values, predictions):
        true_values_without_null = []
        predictions_without_null = []
        for i in range(len(true_values)):
            if true_values[i] != 'O':
                true_values_without_null.append(true_values[i])
                predictions_without_null.append(predictions[i])
        return true_values_without_null, predictions_without_null

    def save2conll(self, markup, path):
        with open(path, 'w') as outf:
            for s in markup:
                for t in s:
                    outf.write(t[0] + ' ' + t[1] + ' ' + '\n')
                outf.write('\n')
        outf.close()

    def save2ann(self, markup, path):
        with open(path, 'w') as outf:
            for index, el in enumerate(markup):
                outf.write('T%s\t' % (index+1,) + el)
        outf.close()

    def iob2spans(self, sequence, strict_iob2=False):
        """
        Turn a sequence of IOB chunks into single tokens
        convert to iob to span
        """

        iobtype = 2 if strict_iob2 else 1
        chunks = []
        spans = []
        current = None
        for i, y in enumerate(sequence):
            label = sequence[i][1]
            if label.startswith('B-'):
                if current is not None:
                    chunks.append('@'.join(current))
                current = [label.replace('B-', ''), '%d' % i]
            elif label.startswith('I-'):
                if current is not None:
                    base = label.replace('I-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                    else:
                        chunks.append('@'.join(current))
                        if iobtype == 2:
                            print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (
                                label, current[0], i))
                        current = [base, '%d' % i]
                else:
                    current = [label.replace('I-', ''), '%d' % i]
                    if iobtype == 2:
                        print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
            else:
                if current is not None:
                    chunks.append('@'.join(current))
                current = None
        if current is not None:
            chunks.append('@'.join(current))
        chunks = set(chunks)

        if chunks:
            for el in chunks:
                chunk_start = int(el.split('@')[1])
                chunk_end = int(el.split('@')[-1])
                spans.append(sequence[chunk_start:chunk_end+1])

        if spans:
            spans = self.form_entity_str(spans)

        return spans

    def form_entity_str(self, spans):
        result = []
        _without_spaces = ['CORREO_ELECTRONICO', 'FECHAS']
        for sp in spans:
            print(sp)
            _str = ''
            _class = sp[0][1].split('-')[-1]
            _start_id = sp[0][2][0]
            _end_id = sp[-1][2][-1]
            for el in sp:
                if el[0] in string.punctuation:
                    _str += el[0]
                elif el[1].split('-')[-1] in _without_spaces:
                    _str += el[0]
                else:
                    _str += ' ' + el[0]
            str_result = '%s %s %s %s\n' % (_class, _start_id, _end_id, _str)
            result.append(str_result)
        return result

    def read_file(self, path2data_file):
        file_object = open(path2data_file, "r")
        data = [list(y) for x, y in itertools.groupby([line for line in file_object.readlines()], lambda z: z == '\n')]
        data = [[t.replace('\n', '') for t in s] for s in data]
        data = [[t.split() for t in s if len(t.split()) > 1] for s in data if s != ['']]
        return data
