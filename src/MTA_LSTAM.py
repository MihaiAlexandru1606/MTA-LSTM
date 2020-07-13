import os
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import layers
import json


class AttentionMTA(layers.Layer):

    def __init__(self, units):
        super(AttentionMTA, self).__init__()
        self.Ua = layers.Dense(units)
        self.Wa = layers.Dense(units)
        self.Va = layers.Dense(1)

    def call(self, inputs, **kwargs):
        coverage_vector, hidden_state, topics_embedding = inputs
        hidden_state = tf.expand_dims(hidden_state, axis=1)

        # print(coverage_vector.shape)  # (batch_size, number_topics)
        # print(hidden_state.shape) # (batch_size, 1, units)
        # print(topics_embedding.shape) # (batch_size, number_topics, embedding_size)

        v = self.Va(tf.nn.tanh(self.Wa(hidden_state) + self.Ua(topics_embedding)))
        g = coverage_vector * tf.reshape(v, [v.shape[0], v.shape[1]])
        alpha = tf.nn.softmax(g, axis=1)

        context = tf.expand_dims(alpha, axis=2) * topics_embedding
        context = tf.reduce_sum(context, axis=1)

        return alpha, context


class MTA_LSTM(tf.keras.Model):

    def __init__(self, units, vocab_size, embedding_size, number_topics, batch_size, dropout=0.2):
        super(MTA_LSTM, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.number_topics = number_topics
        self.embedding = layers.Embedding(vocab_size, embedding_size)
        self.attention = AttentionMTA(units)
        self.Uf = layers.Dense(number_topics * embedding_size)
        self.lstm = layers.GRU(units, return_sequences=True, return_state=True)
        self.softmax = layers.Dense(vocab_size, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None, mask=None):
        word_input, hidden_state, coverage_vector, topics = inputs
        word_input_embedding = self.embedding(word_input)
        topics_embedding = self.embedding(topics)

        alpha, context = self.attention([coverage_vector, hidden_state, topics_embedding])

        # compute output
        x = tf.concat([tf.expand_dims(context, axis=1), word_input_embedding], axis=-1)
        output, hidden_state = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.dropout(output)
        output = self.softmax(output)

        # update coverage vector
        phi = tf.nn.sigmoid(self.Uf(topics_embedding))
        phi = tf.reduce_sum(phi, axis=2)
        coverage_vector = coverage_vector - alpha / phi

        return output, hidden_state, coverage_vector

    def init_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))

    def init_coverage_vector(self):
        return tf.ones((self.batch_size, self.number_topics))
