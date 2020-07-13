import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.mta.proccesing_dataset import get_data_set_mta
from sklearn.model_selection import train_test_split
import numpy as np
from math import log
import time
import json

from src.mta.MTA_LSTAM import MTA_LSTM
from src.mta.proccesing_dataset import get_data_set_mta, reshape_dataset, preprocess_sentence

from src.eval.usage import get_print_mistake, eval_essay


class MTA_Wrapper(object):

    def __init__(self, path_config_file="../../config/mta/config_mta_ro.json"):
        with open(path_config_file, 'r') as read_data:
            config = json.load(read_data)

        self.units = config["units"]
        self.embedding_size = config["embedding_size"]
        self.number_topics = config["number_topics"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.dropout = config["dropout"]
        self.start_seq = config["start_seq"]
        self.end_seq = config["end_seq"]
        self.path_dataset = config["path_dataset"]
        self.steps_to_save = config["steps_to_save"]
        self.checkpoint_dir = config["check_point_save"]
        self.path_save_model = config["path_save_model"]

        essay_tensor, essay_lang_tokenizer, topics_tensor, topics_lang_tokenizer = get_data_set_mta(self.path_dataset,
                                                                                                    self.number_topics)

        self.essay_tensor = essay_tensor
        self.essay_lang_tokenizer = essay_lang_tokenizer
        self.topics_tensor = topics_tensor
        self.topics_lang_tokenizer = topics_lang_tokenizer

        self.vocab_size = len(essay_lang_tokenizer.word_index) + 1
        self.lstm_mta = MTA_LSTM(units=self.units, vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                 number_topics=self.number_topics, batch_size=self.batch_size)
        buffer_size = len(essay_tensor)
        dataset = tf.data.Dataset.from_tensor_slices((essay_tensor, topics_tensor)).shuffle(buffer_size)
        self.dataset = dataset.batch(self.batch_size, drop_remainder=True)
        self.steps_per_epoch = len(essay_tensor) // self.batch_size

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # check_point
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, lstm_mta=self.lstm_mta)

    def _loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, essay, init_hidden, init_coverage_vector, topics):
        loss = 0

        with tf.GradientTape() as tape:
            hidden = init_hidden
            coverage_vector = init_coverage_vector

            # Teacher forcing - feeding the target as the next input
            for time_step in range(1, essay.shape[1]):
                # nu uita sa expandezi dimnsiunea pentru input

                inputs = [tf.expand_dims(essay[:, time_step - 1], axis=1), hidden, coverage_vector, topics]
                predictions, hidden, coverage_vector = self.lstm_mta(inputs)

                loss += self._loss_function(essay[:, time_step], predictions)

        batch_loss = (loss / int(essay.shape[1] - 1))
        variables = self.lstm_mta.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self):

        for epoch in range(self.epochs):
            init_coverage_vector = self.lstm_mta.init_coverage_vector()
            init_hidden_state = self.lstm_mta.init_hidden_state()

            total_loss = 0

            start = time.time()
            for (batch, (essay, topics)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_step(essay, init_hidden_state, init_coverage_vector, topics)

                total_loss += batch_loss

                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

            if (epoch + 1) % self.steps_to_save == 0:
                print("save form ", epoch)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def _init_predict(self, topics):
        topics = preprocess_sentence(topics)
        topics = topics.split(" ")[1:-1]

        topics = [self.topics_lang_tokenizer.word_index[topic] for topic in topics]
        topics = tf.keras.preprocessing.sequence.pad_sequences([topics], maxlen=self.number_topics, padding='post')

        # acum batch_size este 1
        coverage_vector = tf.ones((1, self.number_topics))  # (batch_size, number_topics)
        hidden_state = tf.zeros((1, self.units))  # (batch_size, units)
        dec_input = tf.expand_dims([self.essay_lang_tokenizer.word_index[self.start_seq]], 0)

        return topics, coverage_vector, hidden_state, dec_input

    def predict_greedy(self, topics, max_len_predict=50):
        topics, coverage_vector, hidden_state, dec_input = self._init_predict(topics)
        seq = []

        for _ in range(max_len_predict):
            inputs = (dec_input, hidden_state, coverage_vector, topics)
            predictions, hidden_state, coverage_vector = self.lstm_mta(inputs)

            predicted_id = tf.argmax(predictions[0]).numpy()
            dec_input = tf.expand_dims([predicted_id], 0)
            seq += [self.essay_lang_tokenizer.index_word[predicted_id]]

            if seq[-1] == self.end_seq:
                break

        return " ".join(seq)

    def predict_beam_search(self, topics, beam_width=5, max_len_predict=50):
        topics, coverage_vector, hidden_state, dec_input = self._init_predict(topics)
        # log_prob, seq index, seq string, coverage_vector, hidden_state
        beam_seqs = [(0.0, dec_input, [], coverage_vector, hidden_state)]

        for _ in range(max_len_predict):
            new_beam_seqs = []

            for log_prob, dec_input, seq_string, coverage_vector, hidden_state in beam_seqs:
                inputs = (dec_input, hidden_state, coverage_vector, topics)
                predictions, hidden_state, coverage_vector = self.lstm_mta(inputs)

                top_values, top_indices = tf.math.top_k(predictions[0], k=beam_width)

                top_values = top_values.numpy()
                top_indices = top_indices.numpy()

                for i in range(beam_width):
                    new_dec_input = tf.expand_dims([top_indices[i]], 0)
                    new_seq = seq_string + [self.essay_lang_tokenizer.index_word[top_indices[i]]]

                    if new_seq[-1] == self.end_seq:
                        return " ".join(new_seq)

                    new_log_prob = log_prob - log(top_values[i])
                    new_beam_seqs += [(new_log_prob, new_dec_input, new_seq, coverage_vector, hidden_state)]

                ordered = sorted(new_beam_seqs, key=lambda tup: tup[0])

                beam_seqs = ordered[:beam_width]
                # print(beam_seqs[0][0], beam_seqs[1][0])
        return " ".join(beam_seqs[0][2])

    def predict_sampling(self, topics, k=10, temperature=0.7, max_len_predict=50):
        topics, coverage_vector, hidden_state, dec_input = self._init_predict(topics)
        seq = []
        tf.random.set_seed(0)
        np.random.seed(0)
        for _ in range(max_len_predict):
            inputs = (dec_input, hidden_state, coverage_vector, topics)
            predictions, hidden_state, coverage_vector = self.lstm_mta(inputs)

            top_values, top_indices = tf.math.top_k(predictions[0], k=k)
            top_values = tf.nn.softmax(top_values / temperature)
            top_values = top_values.numpy()
            top_indices = top_indices.numpy()

            predicted_id = np.random.choice(top_indices, p=top_values)

            dec_input = tf.expand_dims([predicted_id], 0)
            seq += [self.essay_lang_tokenizer.index_word[predicted_id]]

            if seq[-1] == self.end_seq:
                break

        return " ".join(seq)

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save_model))

    def save_model(self):
        self.lstm_mta.save_weights(self.path_save_model + "lstm_mta.h5")

    def load_model(self):
        self.lstm_mta.load_weights(self.path_save_model + "lstm_mta.h5")

