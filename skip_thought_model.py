#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.client import device_lib

from log_factory import FinalLogger


class SkipThoughtModel(object):
    """
    Model skip-thought
    """
    VOCAB_SIZE_THRESHOLD_CPU = 20000
    MAX_GRADIENT_NORM = 5.0
    LOG_FILE = 'skip_thought_model.log'

    def __init__(self,
                 vocab_size,
                 start_vocab,
                 max_target_len,
                 unit_type,
                 num_units,
                 num_layers,
                 dropout,
                 embedding_size,
                 learning_rate,
                 num_keep_ckpts):
        self.vocab_size = vocab_size  # src & tgt share vocab_size
        self.start_vocab = start_vocab  # start_vocab = ['<pad>', '<go>', '<eos>', '<unk>']
        self.max_target_len = max_target_len
        # net-parameters
        self.unit_type = unit_type
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.embedding_share = None
        # net-output-data
        self.curr_encoder_output = None
        self.curr_encoder_state = None
        self.prev_train_logits = None
        self.prev_predict_logits = None
        self.next_train_logits = None
        self.next_predict_logits = None
        self.loss = None
        self.gradients = None
        self.train_op = None
        # net-transit-data
        self.encoder_output = None
        self.encoder_state = None
        self.prev_train_decoder_output = None
        self.prev_predict_decoder_output = None
        self.next_train_decoder_output = None
        self.next_predict_decoder_output = None
        # init-log
        self._logger = FinalLogger(self.LOG_FILE)
        # init-device
        self.num_gpus = 0
        self._init_device_gpus()
        # init placeholder
        self._init_placeholder()
        # embeded init
        self._init_embeddings()
        # build graph
        self._build_graph()
        # compute and apply gradients
        self._build_train()
        # predict
        self._build_predict()
        # save train
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=num_keep_ckpts)

    def _init_device_gpus(self):
        """Init device GPU and CPU."""
        gpu_names = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        self.num_gpus = len(gpu_names)
        self._logger.info('{0} GPUs are detected : {1}'.format(self.num_gpus, gpu_names))

    def _init_placeholder(self):
        """Init prev_curr_next data placeholder."""
        self._logger.info('Init prev_curr_next data placeholder.')
        with tf.variable_scope('placeholders'):
            # curr input
            self.curr_source_data = tf.placeholder(tf.int32, [None, None], name='curr_data')
            self.curr_source_seq_len = tf.placeholder(tf.int32, [None], name='curr_data_seq_len')
            self.batch_size = tf.size(self.curr_source_seq_len, name='batch_size')
            # prev target
            self.prev_target_data_input = tf.placeholder(tf.int32, [None, None], name='prev_targets_input')
            self.prev_target_data_output = tf.placeholder(tf.int32, [None, None], name='prev_targets_output')
            self.prev_target_mask = tf.placeholder(tf.float32, [None, None], name='prev_targets_mask')
            self.prev_target_seq_len = tf.placeholder(tf.int32, [None], name='prev_targets_seq_len')
            # next target
            self.next_target_data_input = tf.placeholder(tf.int32, [None, None], name='next_targets_input')
            self.next_target_data_output = tf.placeholder(tf.int32, [None, None], name='next_targets_output')
            self.next_target_mask = tf.placeholder(tf.float32, [None, None], name='next_targets_mask')
            self.next_target_seq_len = tf.placeholder(tf.int32, [None], name='next_targets_seq_len')

    def _build_cell(self, unit_type, num_units, num_layers, dropout):
        """Build cell"""
        cell_list = []
        for i in range(num_layers):
            single_cell = self._create_rnn_cell(
                unit_type=unit_type,
                num_units=num_units,
                dropout=dropout,
                device_str=self._get_device_str(i, self.num_gpus)
            )
            cell_list.append(single_cell)

        if len(cell_list) == 1:
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _create_rnn_cell(self, unit_type, num_units, dropout, device_str=None):
        """Create rnn single-cell"""
        # cell
        if unit_type == 'lstm':
            single_cell = tf.contrib.rnn.LSTMCell(
                num_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        elif unit_type == 'gru':
            single_cell = tf.contrib.rnn.GRUCell(num_units)
        else:
            raise ValueError('Unknown cell type %s!' % unit_type)
        # dropout wrapper
        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout), output_keep_prob=1.0)
        # device wrapper
        if device_str:
            single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
            self._logger.info('  %s, device=%s' % (type(single_cell).__name__, device_str))

        return single_cell

    @staticmethod
    def _get_device_str(device_id, num_gpus):
        """Return a device string for multi-GPU setup."""
        if num_gpus == 0:
            return '/cpu:0'
        device_str_output = '/gpu:%d' % (device_id % num_gpus)
        return device_str_output

    def _get_embed_device(self, vocab_size):
        """Get embed device"""
        if vocab_size < self.VOCAB_SIZE_THRESHOLD_CPU and self.num_gpus > 0:
            return '/gpu:0'
        else:
            return '/cpu:0'

    def _init_embeddings(self):
        """Init embedding."""
        # share vocab
        self._logger.info('Init embedding src_tgt_share.')
        with tf.device(self._get_embed_device(self.vocab_size)):
            self.embedding_share = tf.get_variable(
                'embedding_share', [self.vocab_size, self.embedding_size], dtype=tf.float32)
        self._logger.info(
            '  %s, device=%s' % (type(self.embedding_share).__name__, self._get_embed_device(self.vocab_size)))

    def _build_encoder(self, enc_scope_name):
        """Network encoder."""
        self._logger.info('Build encoder.')
        with tf.variable_scope(enc_scope_name):
            # shape, [batch_size, max_time, embed_size]
            # encoder_embed_input = tf.contrib.layers.embed_sequence(self.curr_source_data, self.vocab_size,
            #                                                        self.embedding_size)
            encoder_embed_input = tf.nn.embedding_lookup(self.embedding_share, self.curr_source_data)
            cell = self._build_cell(self.unit_type, self.num_units, self.num_layers, self.dropout)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(
                cell, encoder_embed_input, sequence_length=self.curr_source_seq_len, dtype=tf.float32)

        return encoder_output, encoder_state

    def _build_decoder(self, encoder_output, encoder_state, target_data, target_seq_len, dec_scope_name):
        """Network decoder."""
        self._logger.info('Build %s.', dec_scope_name)
        with tf.variable_scope(dec_scope_name):
            # decoder_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size]))

            cell = self._build_cell(self.unit_type, self.num_units, self.num_layers, self.dropout)

            # attention-model
            cell, encoder_state = self._build_attention(
                encoder_output, encoder_state, cell)
            # output_layer
            output_layer = Dense(
                self.vocab_size, use_bias=False,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            self._logger.info(' Build decoder train.')
            with tf.variable_scope(dec_scope_name + '_train'):
                # Data format of target_data: <GO>...<PAD>
                # shape: [batch_size, max_time, embed_size], type: float32.
                decoder_embed_input = tf.nn.embedding_lookup(
                    self.embedding_share, target_data)
                train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=decoder_embed_input, sequence_length=target_seq_len, time_major=False)
                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, train_helper, encoder_state, output_layer)
                train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    train_decoder, impute_finished=True, maximum_iterations=self.max_target_len)

            self._logger.info(' Build decoder predict.')
            with tf.variable_scope(dec_scope_name + '_predict', reuse=True):
                # start_tokens = tf.tile(
                #     tf.constant([self.start_vocab.index('<go>')], dtype=tf.int32),
                #     [self.batch_size], name='start_tokens')
                predict_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding_share,
                    tf.fill([self.batch_size], self.start_vocab.index('<go>')),
                    self.start_vocab.index('<eos>'))
                predict_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, predict_helper, encoder_state, output_layer)
                predict_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    predict_decoder, impute_finished=True, maximum_iterations=self.max_target_len)

        return train_decoder_output, predict_decoder_output

    def _build_attention(self, encoder_output, encoder_state, cell):
        """Attention"""
        # attention_states: [batch_size, max_time, num_units]
        # attention_states = tf.transpose(encoder_output, [1, 0, 2])
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.num_units, encoder_output, memory_sequence_length=self.curr_source_seq_len)

        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism, attention_layer_size=self.num_units)

        decoder_initial_state = cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state)

        cell = tf.contrib.rnn.DeviceWrapper(
            cell, self._get_device_str(self.num_layers - 1, self.num_gpus))

        return cell, decoder_initial_state

    def _build_graph(self):
        """Build skip-thought model by seq2seq model"""
        self._logger.info('Build graph.')
        # curr_data encoder
        self.encoder_output, self.encoder_state = self._build_encoder('encoder')
        # prev_data decoder
        self.prev_train_decoder_output, self.prev_predict_decoder_output = self._build_decoder(
            self.encoder_output, self.encoder_state,
            self.prev_target_data_input, self.prev_target_seq_len, 'prev_decoder')
        # next_data decoder
        self.next_train_decoder_output, self.next_predict_decoder_output = self._build_decoder(
            self.encoder_output, self.encoder_state,
            self.next_target_data_input, self.next_target_seq_len, 'next_decoder')
        self._logger.info('Compute loss.')
        # compute loss
        with tf.device(self._get_device_str(self.num_layers - 1, self.num_gpus)):
            # prev loss
            prev_train_logits = tf.identity(
                self.prev_train_decoder_output.rnn_output, name='prev_logits')
            prev_loss = self._compute_loss(
                self.prev_target_data_output, self.prev_target_mask, prev_train_logits)
            # next loss
            next_train_logits = tf.identity(
                self.next_train_decoder_output.rnn_output, name='next_logits')
            next_loss = self._compute_loss(
                self.next_target_data_output, self.next_target_mask, next_train_logits)
            # loss
            self.loss = prev_loss + next_loss

    def _compute_loss(self, target_output, target_mask, logits):
        """Compute optimization loss."""
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        loss = tf.reduce_sum(
            crossent * target_mask) / tf.to_float(self.batch_size)

        return loss

    def _build_train(self):
        """Train, compute and apply gradients"""
        self._logger.info('Build train.')
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_grads, _ = tf.clip_by_global_norm(
            gradients, self.MAX_GRADIENT_NORM)

        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = opt.apply_gradients(
            zip(clipped_grads, params))

    def _build_predict(self):
        """Predict output: curr_data encoder, prev_predict and next_predict data"""
        self._logger.info('Build predict.')
        with tf.device(self._get_device_str(self.num_layers - 1, self.num_gpus)):
            with tf.variable_scope('prev'):
                self.prev_train_logits = tf.identity(
                    self.prev_train_decoder_output.rnn_output, name='logits')
                self.prev_predict_logits = tf.identity(
                    self.prev_predict_decoder_output.sample_id, name='predictions')

            with tf.variable_scope('next'):
                self.next_train_logits = tf.identity(
                    self.next_train_decoder_output.rnn_output, name='logits')
                self.next_predict_logits = tf.identity(
                    self.next_predict_decoder_output.sample_id, name='predictions')

            with tf.variable_scope('curr'):
                self.curr_encoder_output = tf.identity(self.encoder_output, name='output')
                self.curr_encoder_state = tf.identity(self.encoder_state, name='state')
