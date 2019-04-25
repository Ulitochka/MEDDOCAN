import math
import itertools
import pickle
from collections import Counter

from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras.utils import to_categorical
from gensim.models import KeyedVectors


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
bin_models_files = os.path.join(project_path + '/bin/')
w2v_model = '/mnt/storage/Data/cc.es.300.vec'   # cc.es.300.bin

LEN_THRESHOLD = 100
COUNT_LONG_SENT = 0
RANDOM_CHAR_VECTOR_SIZE = 300


data_set = {
    "train": {
        "x": None,
        "y": None,
        "data": [],
        "data_path": os.path.join(project_path + '/data/markups/train/')
    },

    "valid": {
        "x": None,
        "y": None,
        "data": [],
        "data_path": os.path.join(project_path + '/data/markups/dev/')
    }
}


def read_file(path2data_file):
    file_object = open(path2data_file, "r")
    data = [list(y) for x, y in itertools.groupby([line for line in file_object.readlines()], lambda z: z == '\n')]
    data = [[t.replace('\n', '') for t in s] for s in data]
    data = [[t.split() for t in s if len(t.split()) > 1] for s in data if s != ['']]
    return data


for ds in data_set:
    data = [
        read_file(os.path.join(data_set[ds]["data_path"], files)) for files in os.listdir(data_set[ds]["data_path"])
    ]
    data_long = [s for f in data for s in f if len(s) > 50]
    print('Count long sentences: ', len(data_long))
    data_long = [t for s in data_long for t in s]
    data_short = [s for f in data for s in f if len(s) <= 50]
    data_long = [list(y) for x, y in itertools.groupby([line for line in data_long], lambda z: z == ['.', 'O'])]
    data_long = [s for s in data_long if s != [['.', 'O']]]
    data_long = [s + [['.', 'O']] for s in data_long]
    print('Count long sentences after splitting: ', len(data_long))

    for s in data_long:
        if len(s) > LEN_THRESHOLD:
            COUNT_LONG_SENT += 1

    data = data_long + data_short
    data = [el for el in data if el]
    data_set[ds]["x"] = [[t[0] for t in s] for s in data]
    data_set[ds]["y"] = [[t[-1] for t in s] for s in data]

train = data_set["train"]
dev = data_set["valid"]

print('Count long sent in all data:', COUNT_LONG_SENT)

unique_words = sorted(set([w for s in train["x"] + dev["x"] for w in s]))
unique_chars = sorted(set([ch for s in train["x"] + dev["x"] for w in s for ch in w]))
print('unique_chars: ', len(unique_chars))

words2i = {w: i + 1 for i, w in enumerate(unique_words)}
chars2i = {ch: i + 1 for i, ch in enumerate(unique_chars)}
unique_labels = sorted(set([w for s in train["y"] + dev["y"] for w in s]))

labels2ind = {l: i + 1 for i, l in enumerate(unique_labels)}
ind2labels = {i + 1: l for i, l in enumerate(unique_labels)}
print('unique_labels: ', len(labels2ind))

x_train = [[words2i.get(w) for w in s] for s in train["x"]]
y_train = [[labels2ind.get(w) for w in s] for s in train["y"]]

x_test = [[words2i.get(w) for w in s] for s in dev["x"]]
y_test = [[labels2ind.get(w) for w in s] for s in dev["y"]]

max_w_len = max([len(w) for s in train["x"] + dev["x"] for w in s])
print('max_token_len: ', max_w_len)

sentences_lens = [len(s) for s in train["x"] + dev["x"]]
average_len = sum(sentences_lens) // len(sentences_lens)
max_s_len = LEN_THRESHOLD
print('max_sent_len: ', max_s_len)
print('average_sent_len: ', average_len)
print(Counter(sentences_lens))

x_train = pad_sequences(x_train, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
y_train = pad_sequences(y_train, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
y_train = to_categorical(y_train, len(labels2ind)+1)

x_train_chars = [[[chars2i.get(ch) for ch in t] for t in s] for s in train["x"]]
x_train_chars = [pad_sequences(s, maxlen=max_w_len, value=0, padding='pre', truncating='pre') for s in x_train_chars]
x_train_chars = pad_sequences(x_train_chars, maxlen=max_s_len, value=0, padding='pre', truncating='pre')

x_test_chars = [[[chars2i.get(ch) for ch in t] for t in s] for s in dev["x"]]
x_test_chars = [pad_sequences(s, maxlen=max_w_len, value=0, padding='pre', truncating='pre') for s in x_test_chars]
x_test_chars = pad_sequences(x_test_chars, maxlen=max_s_len, value=0, padding='pre', truncating='pre')

x_test = pad_sequences(x_test, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
y_test = pad_sequences(y_test, maxlen=max_s_len, value=0, padding='pre', truncating='pre')
y_test = to_categorical(y_test, len(labels2ind)+1)

print('x_train:', x_train.shape)
print('x_train_chars:', x_train_chars.shape)
print('y_train:', y_train.shape)

print('x_test:', x_test.shape)
print('x_test_chars:', x_test_chars.shape)
print('y_test:', y_test.shape)

########################################################################################################################


def word2char_one_zero_matrix(unique_symbols):
    embed_vocab = list()
    base_vector = np.zeros(len(unique_symbols) * max_w_len)
    embed_vocab.append(base_vector)
    for token in unique_words:
        features_per_token = np.array([], dtype='int8')
        for index_chars in range(0, max_w_len):
            array_char = np.zeros((len(unique_chars),))
            try:
                array_char[unique_chars.index(token[index_chars])] = 1
                # print(word[index_chars], array_char)
            except IndexError:
                pass
            features_per_token = np.append(features_per_token, array_char)
        embed_vocab.append(features_per_token)
    return np.array(embed_vocab).astype('int8')


def w2v_random_matrix():
    embed_vocab = list()
    base_vector = np.zeros(RANDOM_CHAR_VECTOR_SIZE)
    embed_vocab.append(base_vector)
    for token in unique_words:
        limit = math.sqrt(3.0 / RANDOM_CHAR_VECTOR_SIZE)
        features_per_token = np.random.uniform(-limit, limit, RANDOM_CHAR_VECTOR_SIZE)
        embed_vocab.append(features_per_token)
    return np.array(embed_vocab)


def char_random():
    embed_vocab = list()
    base_vector = np.zeros(RANDOM_CHAR_VECTOR_SIZE)
    embed_vocab.append(base_vector)
    for ch in unique_chars:
        limit = math.sqrt(3.0 / RANDOM_CHAR_VECTOR_SIZE)
        features_per_token = np.random.uniform(-limit, limit, RANDOM_CHAR_VECTOR_SIZE)
        embed_vocab.append(features_per_token)
    return np.array(embed_vocab)


def w2v_matrix():
    w2v_vectors = KeyedVectors.load_word2vec_format(w2v_model, binary=False)
    vectors = []
    cover_voc = 0
    base_vector = np.zeros(RANDOM_CHAR_VECTOR_SIZE)
    vectors.append(base_vector)
    for t in unique_words:
        try:
            vectors.append(w2v_vectors[t])
            cover_voc += 1
        except KeyError:
            limit = math.sqrt(3.0 / RANDOM_CHAR_VECTOR_SIZE)
            embedding_matrix = np.random.uniform(-limit, limit, RANDOM_CHAR_VECTOR_SIZE)
            vectors.append(embedding_matrix)
    vectors = np.array(vectors)
    print('create matrix: %s; cover_voc: %s' % (vectors.shape, cover_voc))
    return vectors

#######################################################################################################################


one_zero_matrix = word2char_one_zero_matrix(unique_chars)
w2v_random_matrix = w2v_random_matrix()
w2v_matrix = w2v_matrix()
chars_matrix = char_random()
print("one_zero_matrix: ", one_zero_matrix.shape)
print("random_char_matrix: ", w2v_random_matrix.shape)
print("w2v_matrix: ", w2v_matrix.shape)
print("chars_matrix:", chars_matrix.shape)

with open(os.path.join(project_path + '/data/features/features_nn.pkl'), 'wb') as file:
    pickle.dump({
        'one_zero_matrix': one_zero_matrix,
        'w2v_random_matrix': w2v_random_matrix,
        'w2v_matrix': w2v_matrix,
        "chars_matrix": chars_matrix,

        'unique_words': unique_words,
        'unique_chars': unique_chars,
        'words2i': words2i,
        'chars2i': chars2i,
        'unique_labels': unique_labels,
        'labels2ind': labels2ind,
        'ind2labels': ind2labels,

        'max_s_len': max_s_len,
        'max_w_len': max_w_len,

        'x_train': x_train,
        "x_train_chars": x_train_chars,
        'y_train': y_train,

        'x_test': x_test,
        "x_test_tokens": dev["x"],
        "x_test_chars": x_test_chars,
        'y_test': y_test,

    }, file, protocol=4)
