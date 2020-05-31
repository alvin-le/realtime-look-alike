#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""The main RALM model and related functions."""

import json
import six
import copy
import tensorflow as tf

class DnnConfig(object):
    """Configuration for Youtube DNN Model."""
    
    def __init__(self,
        embedding_size,
        learning_rate=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02):
    
        """
        Constructs DnnConfig.
        Args:
            embedding_size: embedding size
        """
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
    
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a DnnConfig from a Python dictionary of parameters."""
        config = DnnConfig(embedding_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a DnnConfig from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DnnModel(object):
    
    def __init__(self,
               config,
               indexer,
               is_predict,
               uid=None,
               last_cid=None,
               last_gid=None,
               last_cate=None,
               hist_len=None,
               hist_cid=[],
               hist_gid=[],
               hist_cate=[],
               sub_cid=[],
               sub_gid=[],
               sub_cate=[],
               label=[]):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. rue for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
            it is must faster if this is True, on the CPU or GPU, it is faster if
            this is False.
          scope: (optional) variable scope. Defaults to "bert".

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        
        # emb variable
        cid_emb_w = tf.get_variable("cid_emb_w", [indexer.cid_cnt, config.embedding_size], 
            initializer=tf.contrib.layers.xavier_initializer(seed=10))
        cid_b = tf.get_variable("cid_b", [indexer.cid_cnt], initializer=tf.constant_initializer(0.0))
        gid_emb_w = tf.get_variable("gid_emb_w", [indexer.gid_cnt, config.embedding_size],
            initializer=tf.contrib.layers.xavier_initializer(seed=10))
        cate_emb_w = tf.get_variable("cate_emb_w", [indexer.category_cnt, config.embedding_size],
            initializer=tf.contrib.layers.xavier_initializer(seed=10))
        
        # historty seq
        h_cid = tf.nn.embedding_lookup(cid_emb_w, hist_cid)
        h_emb = tf.concat([tf.nn.embedding_lookup(cid_emb_w, hist_cid),
                           tf.nn.embedding_lookup(gid_emb_w, hist_gid),
                           tf.nn.embedding_lookup(cate_emb_w, hist_cate)], axis=-1) # [B,T,3*e]
        
        # historty mask
        hist_len = tf.reshape(hist_len, [tf.shape(hist_len)[0]])
        mask_seq = tf.sequence_mask(hist_len, tf.shape(h_emb)[1], dtype=tf.float32)  # [B,T]
        mask_seq_shape = tf.shape(mask_seq)
        mask = tf.reshape(mask_seq, [mask_seq_shape[0], mask_seq_shape[1]])
        
        mask = tf.expand_dims(mask, -1)  # [B,T,1]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B,T,3*e]

        h_emb *= mask  # [B,T,3*e]
        hist = tf.reduce_sum(h_emb, 1)  # [B,3*e]
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(hist_len, 1), [1, 3 * config.embedding_size]), tf.float32))  # [B,3*e]
        
        # last
        l_emb = tf.concat([tf.nn.embedding_lookup(cid_emb_w, last_cid),
                           tf.nn.embedding_lookup(gid_emb_w, last_gid),
                           tf.nn.embedding_lookup(cate_emb_w, last_cate)], axis=-1)
        l_emb = tf.reshape(l_emb, tf.shape(hist))
        inputs = tf.concat([hist, l_emb], axis=-1)

        # dd net
        layer_1 = tf.layers.dense(inputs, 128, activation=tf.nn.elu, name='layer_dense_1')
        layer_1_bn = tf.layers.batch_normalization(layer_1, name='layer_bn_1')
        layer_2 = tf.layers.dense(layer_1_bn, 3 * config.embedding_size, activation=tf.nn.elu, name='user_vec')
        layer_3 = tf.layers.batch_normalization(layer_2, name='user_vec_bn')
        
        if is_predict:
            self.user_emb = layer_3
            self.out_w = tf.concat([cid_emb_w, cid_emb_w, cid_emb_w], axis=1)
            self.out_w = tf.transpose(self.out_w)
            self.out_b = cid_b

        else:
            self.user_emb = tf.expand_dims(layer_3, 1)  # [B,1,3*e]
            self.out_w = tf.concat([tf.nn.embedding_lookup(cid_emb_w, sub_cid),
                                  tf.nn.embedding_lookup(gid_emb_w, sub_gid),
                                  tf.nn.embedding_lookup(cate_emb_w, sub_cate)], axis=2)  # [B,sample,3*e]
            self.out_w = tf.transpose(self.out_w, perm=[0, 2, 1])  # [B,3*e,sample]
            self.out_b = tf.nn.embedding_lookup(cid_b, sub_cid)  # [B,sample]
    
    def get_user_emb(self):
        return self.user_emb
    
    def get_weight_bias(self):
        return (self.out_w, self.out_b)
