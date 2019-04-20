import os
import pickle
import itertools
from collections import Counter

from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import *
from keras.utils import to_categorical
import keras.initializers
import gc; gc.collect()
from keras import backend as K
from sklearn.metrics import classification_report
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
bin_models_files = os.path.join(project_path + '/bin/')

with open(os.path.join(project_path + '/data/features/features_nn.pkl'), 'rb') as f:
    data = pickle.load(f)

    one_zero_matrix = data['one_zero_matrix']
    random_char_matrix = data['random_char_matrix']
    w2v_matrix = data['w2v_matrix']

    unique_words = data['unique_words']
    unique_chars = data['unique_chars']
    words2i = data['words2i']
    chars2i = data['chars2i']
    unique_labels = data['unique_labels']
    labels2ind = data['labels2ind']
    ind2labels = data['ind2labels']

    max_s_len = data['max_s_len']
    max_w_len = data['max_w_len']

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

########################################################################################################################

LSTM_SIZE = 1024
WDROP_RATE = 0.3

txt_input = Input(shape=(max_s_len,), name='word_input')

txt_embed = Embedding(
    input_dim=len(unique_words) + 1,
    output_dim=w2v_matrix.shape[1],
    input_length=max_s_len,
    weights=[w2v_matrix],
    name='word_embedding',
    trainable=False,
    mask_zero=True
)(txt_input)

########################################################################################################################


input_char_emb = Input((max_s_len,), name='input_char_emb')

char_emb = Embedding(
    input_dim=len(unique_words) + 1,
    output_dim=random_char_matrix.shape[1],
    input_length=max_s_len,
    weights=[random_char_matrix],
    mask_zero=False,
    trainable=False
    )(input_char_emb)

cnn_outputs = []
for el in ((20, 1), (40, 2), (60, 3), (80, 4)):
    cnns = Conv1D(filters=el[0], kernel_size=el[1], padding="same", strides=1)(char_emb)
    cnns = BatchNormalization()(cnns)
    cnns = Activation('tanh')(cnns)
    cnn_outputs.append(cnns)
cnns = concatenate(cnn_outputs, axis=-1, name='cnn_concat')

hway_input = Input(shape=(K.int_shape(cnns)[-1],))
gate_bias_init = keras.initializers.Constant(-2)
transform_gate = Dense(units=K.int_shape(cnns)[-1], bias_initializer=gate_bias_init, activation='sigmoid')(hway_input)
carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnns)[-1],))(transform_gate)
h_transformed = Dense(units=K.int_shape(cnns)[-1])(hway_input)
h_transformed = Activation('relu')(h_transformed)
transformed_gated = Multiply()([transform_gate, h_transformed])
carried_gated = Multiply()([carry_gate, hway_input])
outputs = Add()([transformed_gated, carried_gated])
highway = Model(inputs=hway_input, outputs=outputs)
chars_vectors = highway(cnns)

########################################################################################################################

word_vects = concatenate([cnns, txt_embed], axis=-1, name='concat_word_vectors')

lstm_enc, fh, fc, bh, bc = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, return_state=True))(cnns)
lstm_dec = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True))(lstm_enc, initial_state=[bh, bc, fh, fc])

lyr_crf = CRF(len(labels2ind) + 1)
output = lyr_crf(lstm_dec)

model_mk2 = Model(inputs=[txt_input, input_char_emb], outputs=output)
model_mk2.compile(optimizer='adam', loss=lyr_crf.loss_function)

#######################################################################################################################


early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               mode='min')

cb_nonan = TerminateOnNaN()

model_checkpoint = ModelCheckpoint(filepath=os.path.join(bin_models_files + '/simple_nn_model.pkl'),
                                   monitor='val_loss',
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=True,
                                   mode='min')

cb_redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)

model_mk2.fit(
    x=[x_train, x_train],
    y=y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    verbose=1,
    shuffle=True,
    callbacks=[model_checkpoint, early_stopping, cb_nonan])

#######################################################################################################################


def preparation_data_to_score(yh, pr):
    yh = yh.argmax(2)
    pr = [list(np.argmax(el, axis=1)) for el in pr]
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]

    fyh = [ind2labels.get(c, 'O') for row in yh for c in row]
    fpr = [ind2labels.get(c, 'O') for row in ypr for c in row]
    return fyh, fpr


custom_objects = {'CRF': CRF, 'crf_loss': crf_loss}
model = load_model(os.path.join(bin_models_files + '/simple_nn_model.pkl'), custom_objects=custom_objects)
pr = model.predict(x_test, verbose=1)
y_test, pr = preparation_data_to_score(y_test, pr)
print(classification_report(y_test, pr, digits=4))
