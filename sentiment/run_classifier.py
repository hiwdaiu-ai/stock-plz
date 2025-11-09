# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import argparse
import modeling
import optimization
import tokenization
import tensorflow as tf
import pandas as pd

# Global FLAGS variable will be set in main()
FLAGS = None


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


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
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class SimProcessor(DataProcessor):
  """Processor for the Sim task"""

  # read tsv
  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
    train_data = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[1])
      # text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[0])
      train_data.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return train_data

  # read csv
  # def get_train_examples(self, data_dir):
  #   file_path = os.path.join(data_dir, 'train.csv')
  #   train_df = pd.read_csv(file_path, encoding='utf-8')
  #   train_data = []
  #   for index, train in enumerate(train_df.values):
  #       guid = 'train-%d' % index
  #       text_a = tokenization.convert_to_unicode(str(train[0]))
  #       # text_b = tokenization.convert_to_unicode(str(train[1]))
  #       label = str(train[1])
  #       train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  #   return train_data

  # read txt
  # def get_train_examples(self, data_dir):
  #   file_path = os.path.join(data_dir, 'train_sentiment.txt')
  #   f = open(file_path, 'r')
  #   train_data = []
  #   index = 0
  #   for line in f.readlines():
  #       guid = 'train-%d' % index
  #       line = line.replace("\n", "").split("\t")
  #       text_a = tokenization.convert_to_unicode(str(line[1]))
  #       label = str(line[2])
  #       train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  #       index += 1
  #   return train_data

  # read tsv
  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
    dev_data = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "dev-%d" % (i)
        text_a = tokenization.convert_to_unicode(line[1])
        # text_b = tokenization.convert_to_unicode(line[7])
        label = tokenization.convert_to_unicode(line[0])
        dev_data.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return dev_data

  # # read csv
  # def get_dev_examples(self, data_dir):
  #   file_path = os.path.join(data_dir, 'dev.csv')
  #   dev_df = pd.read_csv(file_path, encoding='utf-8')
  #   dev_data = []
  #   for index, dev in enumerate(dev_df.values):
  #       guid = 'dev-%d' % index
  #       text_a = tokenization.convert_to_unicode(str(dev[0]))
  #       # text_b = tokenization.convert_to_unicode(str(dev[1]))
  #       label = str(dev[1])
  #       dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  #   return dev_data

  # def get_dev_examples(self, data_dir):
  #   file_path = os.path.join(data_dir, 'dev_sentiment.txt')
  #   f = open(file_path, 'r')
  #   dev_data = []
  #   index = 0
  #   for line in f.readlines():
  #       guid = 'dev-%d' % index
  #       line = line.replace("\n", "").split("\t")
  #       text_a = tokenization.convert_to_unicode(str(line[1]))
  #       label = str(line[2])
  #       dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  #       index += 1
  #   return dev_data

  # read tsv
  def get_test_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
    test_data = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "test-%d" % (i)
        text_a = tokenization.convert_to_unicode(line[1])
        # text_b = tokenization.convert_to_unicode(line[7])
        label = tokenization.convert_to_unicode(line[0])
        test_data.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return test_data

  # # read csv
  # def get_test_examples(self, data_dir):
  #   file_path = os.path.join(data_dir, 'test.csv')
  #   test_df = pd.read_csv(file_path, encoding='utf-8')
  #   test_data = []
  #   for index, test in enumerate(test_df.values):
  #       guid = 'test-%d' % index
  #       text_a = tokenization.convert_to_unicode(str(test[0]))
  #       # text_b = tokenization.convert_to_unicode(str(test[1]))
  #       label = str(test[1])
  #       test_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  #   return test_data

  # def get_test_examples(self, data_dir):
  #     file_path = os.path.join(data_dir, 'dev_sentiment.txt')
  #     f = open(file_path, 'r')
  #     test_data = []
  #     index = 0
  #     for line in f.readlines():
  #         guid = 'test-%d' % index
  #         line = line.replace("\n", "").split("\t")
  #         text_a = tokenization.convert_to_unicode(str(line[1]))
  #         label = str(line[2])
  #         test_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  #         index += 1
  #     return test_data

  def get_labels(self):
    return ['0', '1', '2']

class MyTaskProcessor(DataProcessor):
  """Processor for the News data set (GLUE version)."""

  def __init__(self):
      self.labels = ['positive', 'negative', 'nneutral']

  def get_train_examples(self, data_dir):
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
      return self.labels

  def _create_examples(self, lines, set_type):
      """Creates examples for the training and dev sets."""

      examples = []

      for (i, line) in enumerate(lines):
          guid = "%s-%s" % (set_type, i)
          text_a = tokenization.convert_to_unicode(line[1])
          label = tokenization.convert_to_unicode(line[0])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      return examples


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.compat.v1.logging.info("*** Example ***")
    tf.compat.v1.logging.info("guid: %s" % (example.guid))
    tf.compat.v1.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but we need tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=tf.data.AUTOTUNE)
    d = d.batch(batch_size, drop_remainder=drop_remainder)
    d = d.prefetch(tf.data.AUTOTUNE)

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1]

  output_weights = tf.compat.v1.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "output_bias", [num_labels], initializer=tf.compat.v1.zeros_initializer())

  with tf.compat.v1.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, rate=0.1)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      # TPU scaffold removed - using standard checkpoint initialization
      tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.compat.v1.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities})
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main():
  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "sim": SimProcessor,
      "mytask": MyTaskProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.io.gfile.makedirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # GPU/CPU configuration for TensorFlow 2.15 (TPU support removed)
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False,
      use_one_hot_embeddings=False)

  # Use standard Estimator for GPU/CPU training
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=lambda: train_input_fn({'batch_size': FLAGS.train_batch_size}), max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    # TPU-specific padding removed for GPU/CPU training
    
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.compat.v1.logging.info("***** Running evaluation *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(eval_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=lambda: eval_input_fn({'batch_size': FLAGS.eval_batch_size}), steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.io.gfile.GFile(output_eval_file, "w") as writer:
      tf.compat.v1.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    # TPU-specific padding removed for GPU/CPU training
    
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.compat.v1.logging.info("***** Running prediction*****")
    tf.compat.v1.logging.info("  Num examples = %d", len(predict_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=lambda: predict_input_fn({'batch_size': FLAGS.predict_batch_size}))

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.io.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.compat.v1.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # Required parameters
  parser.add_argument("--data_dir", required=True, type=str,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--bert_config_file", required=True, type=str,
                      help="The config json file corresponding to the pre-trained BERT model.")
  parser.add_argument("--task_name", required=True, type=str,
                      help="The name of the task to train.")
  parser.add_argument("--vocab_file", required=True, type=str,
                      help="The vocabulary file that the BERT model was trained on.")
  parser.add_argument("--output_dir", required=True, type=str,
                      help="The output directory where the model checkpoints will be written.")
  
  # Other parameters
  parser.add_argument("--init_checkpoint", default=None, type=str,
                      help="Initial checkpoint (usually from a pre-trained BERT model).")
  parser.add_argument("--do_lower_case", action='store_true',
                      help="Whether to lower case the input text. Should be True for uncased models and False for cased models.")
  parser.add_argument("--max_seq_length", default=300, type=int,
                      help="The maximum total input sequence length after WordPiece tokenization.")
  parser.add_argument("--do_train", action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval", action='store_true',
                      help="Whether to run eval on the dev set.")
  parser.add_argument("--do_predict", action='store_true',
                      help="Whether to run the model in inference mode on the test set.")
  parser.add_argument("--train_batch_size", default=32, type=int,
                      help="Total batch size for training.")
  parser.add_argument("--eval_batch_size", default=8, type=int,
                      help="Total batch size for eval.")
  parser.add_argument("--predict_batch_size", default=8, type=int,
                      help="Total batch size for predict.")
  parser.add_argument("--learning_rate", default=5e-5, type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs", default=3.0, type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion", default=0.1, type=float,
                      help="Proportion of training to perform linear learning rate warmup for.")
  parser.add_argument("--save_checkpoints_steps", default=1000, type=int,
                      help="How often to save the model checkpoint.")
  parser.add_argument("--iterations_per_loop", default=1000, type=int,
                      help="How many steps to make in each estimator call.")
  parser.add_argument("--use_tpu", action='store_true',
                      help="Whether to use TPU or GPU/CPU. (Note: TPU support removed in TF 2.15 migration)")
  
  FLAGS = parser.parse_args()
  
  # Enable TF1 compatibility mode for TF2
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  
  main()
