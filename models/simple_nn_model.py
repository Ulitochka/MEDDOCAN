import pickle
import json

from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import *
import keras.initializers
from keras.activations import softmax, sigmoid, relu
import gc; gc.collect()
from keras import backend as K
from sklearn.metrics import classification_report
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss

CONFIG = dict()
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
bin_models_files = os.path.join(project_path + '/bin/')

with open(os.path.join(project_path + '/data/features/features_nn.pkl'), 'rb') as f:
    data = pickle.load(f)

    one_zero_matrix = data['one_zero_matrix']
    random_char_matrix = data['random_char_matrix']
    w2v_matrix = data['w2v_matrix']
    chars_matrix = data['chars_matrix']

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
    x_train_chars = data['x_train_chars']
    y_train = data['y_train']
    x_test = data['x_test']
    x_test_chars = data['x_test_chars']
    y_test = data['y_test']

CONFIG['max_s_len'] = max_s_len
CONFIG['max_w_len'] = max_w_len
CONFIG['labels2ind'] = labels2ind

########################################################################################################################


def expand_tile(units, axis):
    """
    Expand and tile tensor along given axis
    Args:
        units: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        axis: axis along which expand and tile. Must be 1 or 2
    """
    assert axis in (1, 2)
    n_time_steps = K.int_shape(units)[1]
    repetitions = [1, 1, 1, 1]
    repetitions[axis] = n_time_steps
    if axis == 1:
        expanded = Reshape(target_shape=((1,) + K.int_shape(units)[1:]))(units)
    else:
        expanded = Reshape(target_shape=(K.int_shape(units)[1:2] + (1,) + K.int_shape(units)[2:]))(units)
    return K.tile(expanded, repetitions)


def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """
    Compute additive self attention for time series of vectors (with batch dimension)
            the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
            v is a learnable vector of n_hidden dimensionality,
            W_1 and W_2 are learnable [n_hidden, n_input_features] matrices
    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of2784131 units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output
    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
        """

    n_input_features = K.int_shape(units)[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
    exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
    units_pairs = Concatenate(axis=3)([exp1, exp2])
    query = Dense(n_hidden, activation="tanh")(units_pairs)
    attention = Dense(1, activation=lambda x: softmax(x, axis=2))(query)
    attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
    output = Dense(n_output_features, activation=activation)(attended_units)
    return output


########################################################################################################################

LSTM_SIZE = 1024
WDROP_RATE = 0.3
EPOCHS = 1
BS = 32
INPUT_PROJECTION = False

CONFIG['LSTM_SIZE'] = LSTM_SIZE
CONFIG['WDROP_RATE'] = WDROP_RATE
CONFIG['BS'] = BS
CONFIG['EPOCHS'] = EPOCHS

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

if INPUT_PROJECTION is not None:
    txt_embed = Dense(200, activation='relu')(txt_embed)

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

# char_emb_self_att = additive_self_attention(char_emb, n_hidden=200, n_output_features=200, activation=relu)

########################################################################################################################

cnn_input = Input(shape=(max_s_len, max_w_len), name='cnn_input')

cnn_chars_embed = TimeDistributed(
    Embedding(chars_matrix.shape[0], chars_matrix.shape[1],
              input_length=max_w_len,
              weights=[chars_matrix],
              name='cnn_embedding',
              trainable=True,
              mask_zero=False))(cnn_input)

cnn_chars_outputs = []
for el in ((20, 1), (40, 2), (60, 3), (80, 4)):
    cnns_chars = TimeDistributed(Conv1D(filters=el[0], kernel_size=el[1], padding="same", strides=1))(cnn_chars_embed)
    cnns_chars = TimeDistributed(BatchNormalization())(cnns_chars)
    cnns_chars = TimeDistributed(Activation('tanh'))(cnns_chars)
    cnns_chars = TimeDistributed(GlobalMaxPooling1D(), name='cnn1_gmp')(cnns_chars)
    cnn_chars_outputs.append(cnns_chars)
cnn_chars_outputs = concatenate(cnn_chars_outputs, axis=-1, name='cnn_concat')

hway_input = Input(shape=(K.int_shape(cnn_chars_outputs)[-1],))
gate_bias_init = keras.initializers.Constant(-2)
transform_gate = Dense(units=K.int_shape(cnn_chars_outputs)[-1], bias_initializer=gate_bias_init, activation='sigmoid')(hway_input)
carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnn_chars_outputs)[-1],))(transform_gate)
h_transformed = Dense(units=K.int_shape(cnn_chars_outputs)[-1])(hway_input)
h_transformed = Activation('relu')(h_transformed)
transformed_gated = Multiply()([transform_gate, h_transformed])
carried_gated = Multiply()([carry_gate, hway_input])
outputs = Add()([transformed_gated, carried_gated])
highway = Model(inputs=hway_input, outputs=outputs)
cnn_chars_outputs = TimeDistributed(highway, name='cnn_highway')(cnn_chars_outputs)

########################################################################################################################

word_vects = concatenate([cnns, cnn_chars_outputs, txt_embed], axis=-1, name='concat_word_vectors')

lstm_enc, fh, fc, bh, bc = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, return_state=True))(cnns)
lstm_dec = Bidirectional(LSTM(LSTM_SIZE, return_sequences=True))(lstm_enc, initial_state=[bh, bc, fh, fc])

lyr_crf = CRF(len(labels2ind) + 1)
output = lyr_crf(lstm_dec)

model_mk2 = Model(inputs=[txt_input, input_char_emb, cnn_input], outputs=output)
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
cb_redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model_mk2.fit(
    x=[x_train, x_train, x_train_chars],
    y=y_train,
    batch_size=BS,
    epochs=EPOCHS,
    validation_data=([x_test, x_test, x_test_chars], y_test),
    verbose=1,
    shuffle=True,
    callbacks=[model_checkpoint, cb_redlr, cb_nonan])

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


def remove_null(true_values, predictions):
    true_values_without_null = []
    predictions_without_null = []
    for i in range(len(true_values)):
        if true_values[i] != 'O':
            true_values_without_null.append(true_values[i])
            predictions_without_null.append(predictions[i])
    return true_values_without_null, predictions_without_null


# custom_objects = {'CRF': CRF, 'crf_loss': crf_loss}
# model = load_model(os.path.join(bin_models_files + '/simple_nn_model.pkl'), custom_objects=custom_objects)
test_input = [x_test, x_test, x_test_chars]
pr = model_mk2.predict(test_input, verbose=1)
y_test, pr = preparation_data_to_score(y_test, pr)
y_test, pr = remove_null(y_test, pr)
labels = [el[0] for el in sorted([(el, labels2ind[el]) for el in labels2ind], key= lambda x: x[1])]
clrep = classification_report(y_test, pr, digits=4, target_names=labels)
CONFIG['score'] = clrep
print(clrep)

with open(os.path.join(bin_models_files + '/simple_nn_model_config.json'), 'w') as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=4, separators=(',', ': '))
