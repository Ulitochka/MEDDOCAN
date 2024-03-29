import pickle
import uuid
import json

from tqdm import tqdm
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import *
import keras.initializers
from keras.activations import softmax, sigmoid, relu
from keras.preprocessing.sequence import pad_sequences
import gc;

gc.collect()
from keras import backend as K
from sklearn.metrics import classification_report
from keras_contrib.layers import CRF

from tools.metrics import Metrics
from tools.model_tools import ModelTools


class BaseModel:
    CONFIG = dict()
    LSTM_SIZE = 1024
    DROP_RATE = 0.3
    EPOCHS = 50
    BS = 32
    INPUT_PROJECTION = False
    SELF_ATTENTION = False
    _CRF = True
    OPTIMIZER = {'adam': 'adam', 'adadelta': keras.optimizers.Adadelta(clipnorm=1.)}

    def __init__(self):
        self.project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.bin_models_files = os.path.join(self.project_path + '/bin/')
        self.result_path = os.path.join(self.project_path + '/results/')
        self.dev_set_path = os.path.join(self.project_path + '/data/test/')

        self.experiment_id = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8]

        with open(os.path.join(self.project_path + '/data/features/features_nn.pkl'), 'rb') as f:
            data = pickle.load(f)

            self.one_zero_matrix = data['one_zero_matrix']
            self.w2v_matrix = data['w2v_matrix']
            self.chars_matrix = data['chars_matrix']
            self.gaz_matrix = data['gaz_matrix']

            self.unique_words = data['unique_words']
            self.unique_chars = data['unique_chars']
            self.words2i = data['words2i']
            self.chars2i = data['chars2i']
            self.unique_labels = data['unique_labels']
            self.labels2ind = data['labels2ind']
            self.ind2labels = data['ind2labels']

            self.max_s_len = data['max_s_len']
            self.max_w_len = data['max_w_len']

            self.x_train = data['x_train']
            self.x_train_chars = data['x_train_chars']
            self.y_train = data['y_train']
            self.x_test = data['x_test']
            self.x_test_chars = data['x_test_chars']
            self.y_test = data['y_test']

        self.CONFIG['max_s_len'] = self.max_s_len
        self.CONFIG['max_w_len'] = self.max_w_len
        self.CONFIG['labels2ind'] = self.labels2ind
        self.CONFIG['LSTM_SIZE'] = self.LSTM_SIZE
        self.CONFIG['DROP_RATE'] = self.DROP_RATE
        self.CONFIG['BS'] = self.BS
        self.CONFIG['EPOCHS'] = self.EPOCHS
        self.CONFIG['INPUT_PROJECTION'] = self.INPUT_PROJECTION
        self.CONFIG['SELF_ATTENTION'] = self.SELF_ATTENTION
        self.CONFIG['CRF'] = self._CRF

        self.metrics = Metrics(self.ind2labels)
        self.model_tools = ModelTools()

        with open(os.path.join(
                self.bin_models_files + '/simple_nn_model_config_%s.json' % (self.experiment_id,)), 'w') as f:
            json.dump(self.CONFIG, f, ensure_ascii=False, indent=4, separators=(',', ': '))

        self.model = self.compile_model()

    def expand_tile(self, units, axis):
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

    def additive_self_attention(self, units, n_hidden=None, n_output_features=None, activation=None):
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
        exp1 = Lambda(lambda x: self.expand_tile(x, axis=1))(units)
        exp2 = Lambda(lambda x: self.expand_tile(x, axis=2))(units)
        units_pairs = Concatenate(axis=3)([exp1, exp2])
        query = Dense(n_hidden, activation="tanh")(units_pairs)
        attention = Dense(1, activation=lambda x: softmax(x, axis=2))(query)
        attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
        output = Dense(n_output_features, activation=activation)(attended_units)
        return output

    def multiplicative_self_attention(self, units, n_hidden=None, n_output_features=None, activation=None):
        """
        Compute multiplicative self attention for time series of vectors (with batch dimension)
        the formula: score(h_i, h_j) = <W_1 h_i,  W_2 h_j>,  W_1 and W_2 are learnable matrices
        with dimensionality [n_hidden, n_input_features]
        Args:
            units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
            n_hidden: number of units in hidden representation of similarity measure
            n_output_features: number of features in output dense layer
            activation: activation at the output
        Returns:
            output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]

        Example:
            char_emb_self_att = self.multiplicative_self_attention(
                char_emb,
                n_hidden=None,
                n_output_features=None,
                activation=sigmoid
        )

        """
        n_input_features = K.int_shape(units)[2]
        if n_hidden is None:
            n_hidden = n_input_features
        if n_output_features is None:
            n_output_features = n_input_features
        exp1 = Lambda(lambda x: self.expand_tile(x, axis=1))(units)
        exp2 = Lambda(lambda x: self.expand_tile(x, axis=2))(units)
        queries = Dense(n_hidden)(exp1)
        keys = Dense(n_hidden)(exp2)
        scores = Lambda(lambda x: K.sum(queries * x, axis=3, keepdims=True))(keys)
        attention = Lambda(lambda x: softmax(x, axis=2))(scores)
        mult = Multiply()([attention, exp1])
        attended_units = Lambda(lambda x: K.sum(x, axis=2))(mult)
        output = Dense(n_output_features, activation=activation)(attended_units)
        return output

    def pairwise_mul(self, vests):
        x, y = vests
        return x * y

    def build_embedder(self,
                       matrix,
                       max_seq_len,
                       vocab_size=None,
                       weighted=False,
                       transformed=False,
                       default_dim=300,
                       prefix='word',
                       mask=False,
                       trainable=False,
                       tb=False):
        """
        Построение Embedding layer.
        :param matrix:
        :param max_seq_len:
        :param vocab_size:
        :param weighted:
        :param transformed:
        :param default_dim:
        :param prefix:
        :param mask:
        :param trainable:
        :return:
        """

        inp = Input(shape=(max_seq_len,))

        if not matrix:
            emb_dim = default_dim
            voc_siz = vocab_size
            enc = Embedding(voc_siz, emb_dim, input_length=max_seq_len, mask=mask, trainable=trainable)(inp)
        else:
            emb_dim = matrix.shape[1]
            voc_siz = matrix.shape[0]
            if tb:
                enc = TimeDistributed(
                    Embedding(voc_siz,
                              emb_dim,
                              input_length=max_seq_len,
                              weights=[matrix],
                              mask=mask,
                              trainable=trainable))(inp)
            else:
                enc = Embedding(voc_siz,
                                emb_dim,
                                input_length=max_seq_len,
                                weights=[matrix],
                                mask=mask,
                                trainable=trainable)(inp)

        if transformed:
            trf = Dense(emb_dim)(enc)
            act = LeakyReLU()(trf)
        else:
            act = enc

        if weighted:
            wwt = Embedding(voc_siz,
                            1,
                            input_length=max_seq_len,
                            weights=[np.ones(shape=(voc_siz, 1))])(inp)
            wac = Reshape((-1, 1))(Activation("softmax")(Reshape((-1,))(wwt)))
            out = Lambda(self.pairwise_mul, name='MulLayer')([act, wac])
        else:
            out = act

        return Model(inputs=[inp], outputs=out, name=prefix + '_embedding_model')

    def build_conv1d_dil_block(self, input_shape, prefix="w2v", cnn_options=None):
        """
        https://arxiv.org/pdf/1702.02098.pdf
        https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
        https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f
        https://pdfs.semanticscholar.org/3bb6/3fdb4670745f8c97d8cad1a8a9603b1c16f5.pdf

        Example:
            w2v_conv1d_d_encoder = self.build_conv1d_dil_block(
                w2v_embedder.output_shape,
                prefix="w2v",
                cnn_options=((32, 2, 1), (32, 2, 2), (32, 2, 4), (32, 2, 8), (32, 2, 16), (32, 2, 32), (32, 2, 64)),
            )

        :param input_shape:
        :param prefix:
        :param cnn_options:
        :return:
        """

        inp = Input(shape=tuple(input_shape[-2:]))
        for index_dc, el in enumerate(cnn_options):
            cnns = Conv1D(
                filters=el[0],
                kernel_size=el[1],
                dilation_rate=el[-1],
                padding="same")(inp)
            cnns = BatchNormalization()(cnns)
            cnns = Activation('relu')(cnns)
            inp = cnns
        output = inp
        mod = Model(inputs=inp, outputs=output, name=prefix + '_conv1d_dilated_block')
        return mod

    def build_conv1d_block(self, input_shape, prefix="w2v", cnn_options=None, tb=False):
        """
        Блок свертки.
        :param input_shape:
        :param prefix:
        :param cnn_options:
        :return:
        """

        inp = Input(shape=tuple(input_shape[-2:]))
        output = []
        if not tb:
            for el in cnn_options:
                cnns = Conv1D(
                    filters=el[0],
                    kernel_size=el[1],
                    padding="same",
                    strides=1)(inp)
                cnns = BatchNormalization()(cnns)
                cnns = Activation('tanh')(cnns)
                output.append(cnns)
        else:
            for el in cnn_options:
                cnns_chars = TimeDistributed(Conv1D(filters=el[0], kernel_size=el[1], padding="same", strides=1))(
                    inp)
                cnns_chars = TimeDistributed(BatchNormalization())(cnns_chars)
                cnns_chars = TimeDistributed(Activation('tanh'))(cnns_chars)
                cnns_chars = TimeDistributed(GlobalMaxPooling1D())(cnns_chars)
                output.append(cnns_chars)
        output = concatenate(output, axis=-1)
        mod = Model(inputs=inp, outputs=output, name=prefix + '_conv1d_block')
        return mod

    def build_hw_block(self, input_shape, prefix="w2v"):
        """
        HW-block
        :param input_shape:
        :param prefix:
        :return:
        """

        hway_input = Input(shape=(input_shape[-1],))
        gate_bias_init = keras.initializers.Constant(-2)
        transform_gate = Dense(units=input_shape[-1], bias_initializer=gate_bias_init, activation='sigmoid')(hway_input)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(input_shape[-1],))(transform_gate)
        h_transformed = Dense(units=input_shape[-1])(hway_input)
        h_transformed = Activation('relu')(h_transformed)
        transformed_gated = Multiply()([transform_gate, h_transformed])
        carried_gated = Multiply()([carry_gate, hway_input])
        outputs = Add()([transformed_gated, carried_gated])
        highway_model = Model(inputs=hway_input, outputs=outputs, name=prefix + '_hw_block')
        return highway_model

    def build_aligned_seq2seq_block(self, input):
        """
        Liu & Lane (2016): Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling
        INTERSPEECH 2016, available: https://pdfs.semanticscholar.org/84a9/bc5294dded8d597c9d1c958fe21e4614ff8f.pdf
        :param input:
        :return:
        """

        inp = Input(shape=tuple(K.int_shape(input)[-2:]))
        lstm_enc, fh, fc, bh, bc = Bidirectional(LSTM(self.LSTM_SIZE, return_sequences=True, return_state=True))(inp)
        lstm_enc = Dropout(self.DROP_RATE, name='bidirectional_dropout_enc')(lstm_enc)
        lstm_dec = Bidirectional(LSTM(self.LSTM_SIZE, return_sequences=True))(lstm_enc, initial_state=[bh, bc, fh, fc])
        output = Dropout(self.DROP_RATE, name='bidirectional_dropout_dec')(lstm_dec)
        aligned_seq2seq_model = Model(inputs=inp, outputs=output, name='aligned_seq2seq')
        return aligned_seq2seq_model

    def compile_model(self):

        model = None

        ########################################### GAZ EMD ############################################################

        # gaz_input = Input(shape=(self.max_s_len,))
        #
        # gaz_embed = Embedding(
        #     input_dim=len(self.unique_words) + 1,
        #     output_dim=self.gaz_matrix.shape[1],
        #     input_length=self.max_s_len,
        #     weights=[self.gaz_matrix],
        #     name='gaz_embedding',
        #     trainable=False,
        #     mask_zero=False
        # )(gaz_input)

        ########################################### W2V WORD EMBD ######################################################

        txt_input = Input(shape=(self.max_s_len,))

        # w2v_embedder = self.build_embedder(
        #     matrix=self.w2v_matrix,
        #     max_seq_len=self.max_s_len,
        #     vocab_size=len(self.unique_words) + 1,
        #     prefix='w2v',
        #     mask=True,
        #     trainable=False)

        txt_embed = Embedding(
            input_dim=len(self.unique_words) + 1,
            output_dim=self.w2v_matrix.shape[1],
            input_length=self.max_s_len,
            weights=[self.w2v_matrix],
            name='word_embedding',
            trainable=False,
            mask_zero=False
        )(txt_input)

        ######################################## ONE-HOT CHAR EMBD #####################################################

        input_char_emb = Input((self.max_s_len,))

        char_emb = Embedding(
            input_dim=len(self.unique_words) + 1,
            output_dim=self.one_zero_matrix.shape[1],
            input_length=self.max_s_len,
            weights=[self.one_zero_matrix],
            mask_zero=False,
            trainable=False
        )(input_char_emb)

        # o_h_char_embedder = self.build_embedder(
        #     matrix=self.one_zero_matrix,
        #     max_seq_len=self.max_s_len,
        #     vocab_size=len(self.unique_words) + 1,
        #     prefix='o_h_char',
        #     mask=False,
        #     trainable=False)

        cnn_outputs = []
        for index_c, el in enumerate(((20, 1), (40, 2), (60, 3), (80, 4), (100, 5))):
            cnns = Conv1D(
                filters=el[0],
                kernel_size=el[1],
                padding="same",
                strides=1)(char_emb)
            cnns = BatchNormalization()(cnns)
            cnns = Activation('tanh')(cnns)
            cnn_outputs.append(cnns)
        cnns = concatenate(cnn_outputs, axis=-1, name='cnn_concat')

        # o_h_cnn1d_encoder = self.build_conv1d_block(
        #     o_h_char_embedder.output_shape,
        #     prefix="o_h_char",
        #     cnn_options=((20, 1), (40, 2), (60, 3), (80, 4), (100, 5)))

        # o_h_hw_block = self.build_hw_block(
        #     o_h_cnn1d_encoder.output_shape,
        #     prefix="o_h_char")

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

        ######################################## RANDOM CHAR EMB #######################################################

        cnn_input = Input(shape=(self.max_s_len, self.max_w_len))

        cnn_chars_embed = TimeDistributed(
            Embedding(self.chars_matrix.shape[0], self.chars_matrix.shape[1],
                      input_length=self.max_w_len,
                      weights=[self.chars_matrix],
                      trainable=True,
                      mask_zero=False))(cnn_input)

        # random_char_embedder = self.build_embedder(
        #     matrix=self.one_zero_matrix,
        #     max_seq_len=self.max_s_len,
        #     vocab_size=len(self.unique_words) + 1,
        #     prefix='random_char',
        #     mask=False,
        #     trainable=True,
        #     tb=True)

        cnn_chars_outputs = []
        for el in ((20, 1), (40, 2), (60, 3), (80, 4), (100, 5)):
            cnns_chars = TimeDistributed(Conv1D(filters=el[0], kernel_size=el[1], padding="same", strides=1))(cnn_chars_embed)
            cnns_chars = TimeDistributed(BatchNormalization())(cnns_chars)
            cnns_chars = TimeDistributed(Activation('tanh'))(cnns_chars)
            cnns_chars = TimeDistributed(GlobalMaxPooling1D())(cnns_chars)
            cnn_chars_outputs.append(cnns_chars)
        cnn_chars_outputs = concatenate(cnn_chars_outputs, axis=-1)

        # random_cnn1d_encoder = self.build_conv1d_block(
        #     random_char_embedder.output_shape,
        #     prefix="random_char",
        #     cnn_options=((20, 1), (40, 2), (60, 3), (80, 4), (100, 5)),
        #     tb=True)

        hway_input = Input(shape=(K.int_shape(cnn_chars_outputs)[-1],))
        gate_bias_init = keras.initializers.Constant(-2)
        transform_gate = Dense(units=K.int_shape(cnn_chars_outputs)[-1], bias_initializer=gate_bias_init, activation='sigmoid')(
            hway_input)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnn_chars_outputs)[-1],))(transform_gate)
        h_transformed = Dense(units=K.int_shape(cnn_chars_outputs)[-1])(hway_input)
        h_transformed = Activation('relu')(h_transformed)
        transformed_gated = Multiply()([transform_gate, h_transformed])
        carried_gated = Multiply()([carry_gate, hway_input])
        outputs = Add()([transformed_gated, carried_gated])
        highway = Model(inputs=hway_input, outputs=outputs)
        cnn_chars_outputs = TimeDistributed(highway)(cnn_chars_outputs)

        # random_hw_block = self.build_hw_block(
        #     random_cnn1d_encoder.output_shape,
        #     prefix="random_char")

        ############################################## RNN PART ########################################################

        # aligned_seq2seq_model = self.build_aligned_seq2seq_block(word_vects)

        word_vects = concatenate([chars_vectors, cnn_chars_outputs, txt_embed], axis=-1)

        lstm_enc, fh, fc, bh, bc = Bidirectional(LSTM(self.LSTM_SIZE, return_sequences=True, return_state=True))(
            word_vects)     # cnn
        lstm_enc = Dropout(self.DROP_RATE, name='bidirectional_dropout_enc')(lstm_enc)
        lstm_dec = Bidirectional(LSTM(self.LSTM_SIZE, return_sequences=True))(lstm_enc, initial_state=[bh, bc, fh, fc])
        lstm_dec = Dropout(self.DROP_RATE, name='bidirectional_dropout_dec')(lstm_dec)

        if self._CRF:
            lyr_crf = CRF(len(self.labels2ind) + 1)
            output = lyr_crf(lstm_dec)
            model = Model(inputs=[txt_input, input_char_emb, cnn_input], outputs=output)
            model.compile(optimizer=self.OPTIMIZER['adam'], loss=lyr_crf.loss_function)
            model.summary()
        else:
            output = Dense(len(self.labels2ind) + 1, activation='softmax')(lstm_dec)
            model = Model(inputs=[txt_input, input_char_emb, cnn_input], outputs=output)
            model.compile(optimizer=self.OPTIMIZER['adam'], loss="categorical_crossentropy")
            model.summary()

        return model

    def train(self):

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       verbose=1,
                                       mode='min')

        cb_nonan = TerminateOnNaN()

        cb_redlr = {
            '0': ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1),
            '1': ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001, verbose=1)
        }

        self.model.fit(
            x=[self.x_train, self.x_train, self.x_train_chars],
            y=self.y_train,
            batch_size=self.BS,
            epochs=self.EPOCHS,
            validation_data=([self.x_test, self.x_test, self.x_test_chars], self.y_test),
            verbose=1,
            shuffle=True,
            callbacks=[self.metrics, cb_nonan])

    def feature_extractor(self, sentence):
        t_indexes = [self.words2i.get(w) for w in sentence]
        t_ind_pad = pad_sequences([t_indexes], maxlen=self.max_s_len, value=0, padding='pre', truncating='pre')
        tch_indexes = [[self.chars2i.get(ch) for ch in t] for t in sentence]
        tch_ind_pad = pad_sequences(tch_indexes, maxlen=self.max_w_len, value=0, padding='pre', truncating='pre')
        tch_ind_pad = pad_sequences([tch_ind_pad], maxlen=self.max_s_len, value=0, padding='pre', truncating='pre')
        return [t_ind_pad, t_ind_pad, tch_ind_pad]

    def predict(self, experiment_id):
        self.model.load_weights(os.path.join(self.bin_models_files + '%s.h5' % (experiment_id,)))

        for doc in tqdm(os.listdir(self.dev_set_path)[:5]):
            markups = []
            seqs = self.model_tools.read_file(os.path.join(self.dev_set_path, doc))
            seqs_tokens = [[t[0] for t in s] for s in seqs]
            seqs_range = [[(t[1], t[2]) for t in s] for s in seqs]

            for i_s in range(len(seqs_tokens)):
                sentence = seqs_tokens[i_s]
                sentence = [sentence[i * self.max_s_len:(i + 1) * self.max_s_len] for i in
                            range((len(sentence) + self.max_s_len - 1) // self.max_s_len)]
                labels = []
                for s in sentence:
                    features = self.feature_extractor(s)
                    s_pr = self.model.predict(features)
                    pr = [list(np.argmax(el, axis=1)) for el in s_pr]
                    coords = [np.where(yhh > 0)[0][0] for yhh in features[0]]
                    ypr = [prr[co:] for prr, co in zip(pr, coords)]
                    y_l = [self.ind2labels.get(l, 'O') for l in ypr[0]]
                    labels.extend(y_l)
                mrk = list(zip([t for s in sentence for t in s], labels, seqs_range[i_s]))
                markups.append(mrk)

            spans = [sp for ms in markups for sp in self.model_tools.iob2spans(ms)][::-1]
            self.model_tools.save2ann(spans, os.path.join(self.result_path + '%s.ann' % (doc.split('.')[0],)))
            # self.model_tools.save2conll(markups, os.path.join(self.result_path + '%s' % (doc,)))

    def estimate(self, experiment_id):
        self.model.load_weights(os.path.join(self.bin_models_files + '%s.h5' % (experiment_id,)))
        test_input = [self.x_test, self.x_test, self.x_test_chars]
        pr = self.model.predict(test_input, verbose=1)
        y_test, pr = self.model_tools.preparation_data_to_score(self.y_test, pr, self.ind2labels)
        y_test, pr = self.model_tools.remove_null(y_test, pr)
        labels = [el[0] for el in sorted([(el, self.labels2ind[el]) for el in self.labels2ind], key=lambda x: x[1])]
        clrep = classification_report(y_test, pr, digits=4, target_names=labels)
        print(clrep)


if __name__ == '__main__':
    base_model = BaseModel()
    # base_model.train()
    # base_model.estimate('simple_nn_model_2019_04_28-15_55_43-b3a9abaf')
    base_model.predict(experiment_id='simple_nn_model_2019_04_28-15_55_43-b3a9abaf')
