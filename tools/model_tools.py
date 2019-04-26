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

    def save_to_conll(self, markup, path):
        with open(path, 'w') as outf:
            for s in markup:
                for t in s:
                    outf.write(t[0] + ' ' + t[1] + ' ' + '\n')
                outf.write('\n')
        outf.close()
