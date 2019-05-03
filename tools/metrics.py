import os
import time
import uuid

from keras.callbacks import Callback
from keras_contrib.utils import save_load_utils
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from tools.model_tools import ModelTools


class Metrics(Callback):
    def __init__(self, ind2labels, model_type):
        super().__init__()
        self.ind2labels = ind2labels
        self.model_type = model_type
        self.model_tools = ModelTools()
        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.bin_models_files = os.path.join(project_path + '/bin/')
        self.experiment_id = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8]

    def on_train_begin(self, logs={}):
        self.fmacro_best = 0
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Validation data:
            (6991, 100)
            (6991, 100)
            (6991, 100, 30)
            (6991, 100, 24)
            (6991,)
            0.0
        :param epoch:
        :param logs:
        :return:
        """

        val_data = self.validation_data

        if self.model_type == 2:
            validation_input = [val_data[0], val_data[1], val_data[2], val_data[3]]
            validation_test = val_data[4]
        else:
            validation_input = [val_data[0], val_data[1]]
            validation_test = val_data[2]

        val_predict = self.model.predict(validation_input)

        val_targ, val_predict = self.model_tools.preparation_data_to_score(validation_test, val_predict, self.ind2labels)
        val_targ, val_predict = self.model_tools.remove_null(val_targ, val_predict)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        if _val_f1 > self.fmacro_best:
            self.fmacro_best = _val_f1
            print("New best f1macro: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))

            # save architecture with json
            with open(os.path.join(self.bin_models_files + '/simple_nn_model_%s.json' % (self.experiment_id,)), 'w') as f:
                f.write(self.model.to_json())
            # save weights
            save_load_utils.save_all_weights(
                self.model, os.path.join(self.bin_models_files + '/simple_nn_model_%s.h5' % (self.experiment_id,)))
        else:
            print(
                "Ff1macro: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))

        return
