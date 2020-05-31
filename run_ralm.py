# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import pickle
import datetime
import random
import numpy as np
import tensorflow as tf
import sys
import modeling
import indexing
import optimization
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

flags = tf.flags
FLAGS = flags.FLAGS
## Required parameters
flags.DEFINE_string("data_dir", None,
    "The input data dir.")

flags.DEFINE_string("dnn_config_file", None,
    "The config json file specifies the model architecture.")

flags.DEFINE_string("output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
         "The vocabulary file that the model was trained on.")

flags.DEFINE_bool("do_train", False,
    "Whether to run training.")

flags.DEFINE_bool("do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 64,
    "Total batch size for training.")                              

flags.DEFINE_integer("eval_batch_size", 8,
    "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, 
    "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-3,
    "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
    "Total number of training epochs to perform.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, 
    "How often to save the model checkpoint.")

flags.DEFINE_float("warmup_proportion", 0.2,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained  model).")

flags.DEFINE_integer("iterations_per_loop", 1000,                                                                   
    "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")                                         
tf.flags.DEFINE_integer("num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string('export_dir', 'export', 'export dir or tf serving')

class InputExample(object):
    """A single training/test example for simple classification."""
    def __init__(self, guid, uid, last_item, hist_item, sub_sample=None):
        self.guid = guid
        self.uid = uid
        self.last_item = last_item
        self.hist_item = hist_item
        self.sub_sample = sub_sample

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()
    
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class ClkProcessor(DataProcessor):
    """Processor for the clk behavior data set."""
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
            
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
            
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")
    
    def get_labels(self, data_dir):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if 4 > len(line):
                continue
            guid = "%s-%s" % (set_type, i)
            uid = line[0]
            last_item = line[1]
            hist_item = line[2]
            sub_sample = None
            sub_sample = line[3]
            examples.append(
                InputExample(guid=guid, uid=uid, last_item=last_item, 
                                hist_item=hist_item, sub_sample=sub_sample)) 
            
        return examples


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, uid, 
        last_cid, last_gid, last_category, 
        hist_cid, hist_gid, hist_category,
        sub_cid = None, sub_gid = None, sub_category = None,
        label = None):
    self.uid = uid
    self.last_cid = last_cid
    self.last_gid = last_gid
    self.last_category = last_category
    self.hist_cid = hist_cid
    self.hist_gid = hist_gid
    self.hist_category = hist_category
    self.hist_len = len(hist_cid)
    self.sub_cid = sub_cid
    self.sub_gid = sub_gid
    self.sub_category = sub_category
    self.label = label


def convert_single_example(ex_index, example, indexer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    last_cid, last_gid, last_category = indexer.convert_items_to_ids([example.last_item])
    hist = [ x for x in example.hist_item.strip().split(",")]
    hist_cid, hist_gid, hist_category = indexer.convert_items_to_ids(hist)
    sub_cid = None
    sub_gid = None
    sub_category = None
    label = None
    if example.sub_sample:
        sub = [x for x in example.sub_sample.strip().split(",")]
        sub_cid, sub_gid, sub_category = indexer.convert_items_to_ids(sub)
        label = np.zeros(len(sub), np.int64)
        label[0] = 1

    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("uid: %s" % (example.uid))
        tf.logging.info("last_item: %s" % (example.last_item))
        tf.logging.info("hist_item: %s" % (example.hist_item))
        if example.sub_sample:
            tf.logging.info("sub_sample: %s" % (example.sub_sample))
        
        tf.logging.info("*** Feature ***")
        tf.logging.info("last_item[cid:gid:cate]: %s:%s:%s" % (last_cid[0], last_gid[0], last_category[0]))
        tf.logging.info("hist_item_cid: %s" % ",".join([str(x) for x in hist_cid]))
        tf.logging.info("hist_item_gid: %s" % ",".join([str(x) for x in hist_gid]))
        tf.logging.info("hist_item_cate: %s" % ",".join([str(x) for x in hist_category]))
        if example.sub_sample:
            tf.logging.info("sub_sample_cid: %s" % ",".join([str(x) for x in sub_cid]))
            tf.logging.info("sub_sample_gid: %s" % ",".join([str(x) for x in sub_gid]))
            tf.logging.info("sub_sample_cate: %s" % ",".join([str(x) for x in sub_category]))
            tf.logging.info("label: %s" % ",".join([str(x) for x in label]))

    feature = InputFeatures(
        uid=example.uid,
        last_cid=last_cid[0],
        last_gid=last_gid[0],
        last_category=last_category[0],
        hist_cid=hist_cid,
        hist_gid=hist_gid,
        hist_category=hist_category,
        sub_cid=sub_cid,
        sub_gid=sub_gid,
        sub_category=sub_category,
        label=label)
    return feature


def file_based_convert_examples_to_features(examples, indexer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
        return f
    def create_bytes_feature(values):
        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
        return f
    def create_int_feature_list(values):
        fea = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), values))
        f = tf.train.FeatureList(feature=fea)
        return f
    
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
                   
        feature = convert_single_example(ex_index, example, indexer)    
        
        tf_example = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'uid': create_bytes_feature(feature.uid),
                    'last_cid': create_int_feature(feature.last_cid),
                    'last_gid': create_int_feature(feature.last_gid),
                    'last_category': create_int_feature(feature.last_category),
                    'hist_len': create_int_feature(feature.hist_len)
                }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'hist_cid': create_int_feature_list(feature.hist_cid),
                    'hist_gid': create_int_feature_list(feature.hist_gid),
                    'hist_category': create_int_feature_list(feature.hist_category),
                    'sub_cid': create_int_feature_list(feature.sub_cid),
                    'sub_gid': create_int_feature_list(feature.sub_gid),
                    'sub_category': create_int_feature_list(feature.sub_category),
                    'label': create_int_feature_list(feature.label)
            })
        )
        writer.write(tf_example.SerializeToString())


def input_fn_builder(input_file, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    
    context_features = {
        "uid": tf.FixedLenFeature([], dtype=tf.string),
        "last_cid": tf.FixedLenFeature([], dtype=tf.int64),
        "last_gid": tf.FixedLenFeature([], dtype=tf.int64),
        "last_category": tf.FixedLenFeature([], dtype=tf.int64),
        "hist_len": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "hist_cid": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "hist_gid": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "hist_category": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "sub_cid": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "sub_gid": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "sub_category": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "label": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    def _decode_record(record, context_features, sequence_features):
        """Decodes a record to a TensorFlow example."""
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=record,
                context_features=context_features,
                sequence_features=sequence_features
        )
        example = dict()
        example["uid"] = context_parsed["uid"]
        example["last_cid"] = context_parsed["last_cid"]
        example["last_gid"] = context_parsed["last_gid"]
        example["last_category"] = context_parsed["last_category"]
        example["hist_len"] = context_parsed["hist_len"]
        example["hist_cid"] = sequence_parsed["hist_cid"]
        example["hist_gid"] = sequence_parsed["hist_gid"]
        example["hist_category"] = sequence_parsed["hist_category"]
        example["sub_cid"] = sequence_parsed["sub_cid"]
        example["sub_gid"] = sequence_parsed["sub_gid"]
        example["sub_category"] = sequence_parsed["sub_category"]
        example["label"] = sequence_parsed["label"]
        return example 

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=batch_size * 10)
       
        padded_shapes = {
            "uid":[],
            "last_cid":[],
            "last_gid":[], 
            "last_category":[], 
            "hist_len":[],
            "hist_cid":[None],
            "hist_gid":[None],
            "hist_category":[None],
            "sub_cid":[None],
            "sub_gid":[None],
            "sub_category":[None],
            "label":[None]
        }
        d = tf.data.TFRecordDataset(input_file) \
            .map(lambda record: _decode_record(record, context_features, sequence_features)) \
            .padded_batch(batch_size, padded_shapes=padded_shapes)
        return d.make_one_shot_iterator().get_next()
        
    return input_fn


def create_model(dnn_config, indexer, is_pred, uid, last_cid, last_gid, last_cate, 
        hist_len, hist_cid, hist_gid, hist_cate, sub_cid, sub_gid, sub_cate, label, num_labels):
    """Creates a classification model."""
    model = modeling.DnnModel(config=dnn_config,
        indexer=indexer,
        is_predict=is_pred,
        uid=uid,
        last_cid=last_cid,
        last_gid=last_gid,
        last_cate=last_cate,
        hist_len=hist_len,
        hist_cid=hist_cid,
        hist_gid=hist_gid,
        hist_cate=hist_cate,
        sub_cid=sub_cid,
        sub_gid=sub_gid,
        sub_cate=sub_cate,
        label=label)
        
    user_emb = model.get_user_emb()
    (wei, bias) = model.get_weight_bias()

    with tf.variable_scope("loss"):
        if is_pred:
            logits = tf.matmul(user_emb, wei) + bias
        else:
            logits = tf.squeeze(tf.matmul(user_emb, wei), axis=1) + bias
        
        with tf.control_dependencies([
            tf.Assert(tf.is_numeric_tensor(logits), [logits]),
            tf.assert_non_negative(label, [label])
        ]):
           
            probabilities = tf.nn.softmax(logits)
            per_example_loss = tf.reduce_sum(-tf.cast(label, dtype=tf.float32) * tf.log(probabilities + 1e-24),  axis=-1)
            loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities, user_emb)

def model_fn_builder(dnn_config, indexer, num_labels, init_checkpoint, learning_rate,
                num_train_steps, num_warmup_steps):
    
    """Returns `model_fn` closure for Estimator."""
    
    def model_fn(features, labels, mode, params):
        """The `model_fn` for Estimator."""
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        
        uid = features["uid"]
        last_cid = features["last_cid"]
        last_gid = features["last_cid"]
        last_cate = features["last_category"]
        hist_len = features["hist_len"]
        hist_cid = features["hist_cid"]
        hist_gid = features["hist_gid"]
        hist_cate = features["hist_category"]
        sub_cid = features["sub_cid"]
        sub_gid = features["sub_gid"]
        sub_cate = features["sub_category"]
        label = features["label"]

        is_pred = (mode == tf.estimator.ModeKeys.PREDICT)

        (total_loss, per_example_loss, logits, probabilities, user_emb) = create_model(
            dnn_config, indexer, is_pred, uid, last_cid, last_gid, last_cate, hist_len, 
            hist_cid, hist_gid, hist_cate, sub_cid, sub_gid, sub_cate, label, num_labels)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            global_step = tf.train.get_or_create_global_step()
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            gradients = tf.gradients(total_loss, tvars)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
            train_op = opt.apply_gradients(zip(clip_gradients, tvars), global_step=global_step)
            global_step.assign_add(1)
            
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        
        elif mode == tf.estimator.ModeKeys.EVAL:
            
            def macro_f1(labels, predictions, weights=None, 
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):

                tp = tf.cast(tf.count_nonzero(predictions * labels, axis=0), tf.float32)
                fp = tf.cast(tf.count_nonzero(predictions * (1 - labels), axis=0), tf.float32)
                fn = tf.cast(tf.count_nonzero((1 - predictions) * labels, axis=0), tf.float32)
                precision = 1.0 * tp / (tp + fp + 1e-16)
                recall = 1.0 * tp / (tp + fn + 1e-16)
                f1, f1_op = tf.metrics.mean(2.0 * precision * recall / (precision + recall + 1e-16))
                
                if metrics_collections:
                    ops.add_to_collections(metrics_collections, f1)

                if updates_collections:
                    ops.add_to_collections(updates_collections, f1_op)

                return f1, f1_op


            def metric_fn(per_example_loss, label, probabilities):
                y = tf.cast(label, tf.float32)
                y_pred = tf.cast(tf.greater(probabilities, 0.5), tf.float32)
                precision = tf.metrics.precision(y, y_pred)
                recall = tf.metrics.recall(y, y_pred)
                f1 = macro_f1(y, y_pred)
                auc = tf.metrics.auc(label, probabilities)
                loss = tf.metrics.mean(per_example_loss)
                
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f1": f1,
                    "eval_auc": auc,
                    "eval_loss": loss,
                }
            
            eval_metrics = (metric_fn, [per_example_loss, label, probabilities])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                "probabilities": probabilities,
                "user_id": uid,
                "user_emb": user_emb,
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, 
                    predictions=predictions, 
                    export_outputs=export_outputs, 
                    scaffold_fn=scaffold_fn)
        
        return output_spec

    return model_fn

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    
    if not FLAGS.data_dir:
        raise ValueError("Data dir cannot be empty")
    
    tf.gfile.MakeDirs(FLAGS.output_dir)
    
    dnn_config = modeling.DnnConfig.from_json_file(FLAGS.dnn_config_file)
    
    processor = ClkProcessor()
    
    label_list = processor.get_labels(FLAGS.data_dir)
    
    indexer = indexing.Indexer(vocab_file=FLAGS.vocab_file)
    
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        
    model_fn = model_fn_builder(
        dnn_config=dnn_config,
        indexer=indexer,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
        
        
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(train_examples, indexer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = input_fn_builder(
            input_file=train_file,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        
        # save model
        def serving_input_fn():
            uid = tf.placeholder(tf.string, [None, 1], name='uid')
            last_cid = tf.placeholder(tf.int64, [None, 1], name='last_cid')
            last_gid = tf.placeholder(tf.int64, [None, 1], name='last_gid')
            last_category = tf.placeholder(tf.int64, [None, 1], name='last_category')
            hist_len = tf.placeholder(tf.int64, [None, 1], name='hist_len')
            hist_cid = tf.placeholder(tf.int64, [None, None], name='hist_cid')
            hist_gid = tf.placeholder(tf.int64, [None, None], name='hist_gid')
            hist_category = tf.placeholder(tf.int64, [None, None], name='hist_category')
            sub_cid = tf.placeholder(tf.int64, [None, None], name='sub_cid')
            sub_gid = tf.placeholder(tf.int64, [None, None], name='sub_gid')
            sub_category = tf.placeholder(tf.int64, [None, None], name='sub_category')
            label = tf.placeholder(tf.int64, [None, None], name='label')
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
                 'uid': uid,
                 'last_cid': last_cid,
                 'last_gid': last_gid,
                 'last_category': last_category,
                 'hist_len': hist_len,
                 'hist_cid': hist_cid,
                 'hist_gid': hist_gid,
                 'hist_category': hist_category,
                 'sub_cid': sub_cid,
                 'sub_gid': sub_gid,
                 'sub_category': sub_category,
                 'label': label
                 })()
            return input_fn

        estimator._export_to_tpu = False
        estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)
  
  
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(eval_examples, indexer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_file=eval_file,
            is_training=False,
            drop_remainder=False)
        
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, indexer, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=predict_file,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        
        output_user_vec = os.path.join(FLAGS.output_dir, "user_vec.tsv")
        with tf.gfile.GFile(output_user_vec, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for pred in result:
                output_line = pred["user_id"] + "\t"
                output_line += ",".join(
                    str(prob) for prob in pred["user_emb"]) + "\n"
                writer.write(output_line)
        
        output_item_vec = os.path.join(FLAGS.output_dir, "item_vec.tsv")
        cid_wei = estimator.get_variable_value("cid_emb_w")
        cid_bias = estimator.get_variable_value('cid_b')
        [rows, cols] = cid_wei.shape
        with tf.gfile.GFile(output_item_vec, "w") as writer:
            tf.logging.info("***** Predict item vecs *****")
            for i in range(rows):
                output_line = str(indexer.convert_ids_to_items([i])[0]) + "\t"
                output_line += ",".join(
                    str(x) for x in cid_wei[i,:])
                output_line += (str(cid_bias[i]) + "\n")
                writer.write(output_line)

if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
