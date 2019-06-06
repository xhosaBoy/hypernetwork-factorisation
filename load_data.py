# std
import os
import json
import pickle
import functools
from collections import defaultdict

# 3rd Party
import numpy as np
import tensorflow as tf

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class Data:

    def __init__(self, dirname='data', dataset='FB15k-237', reverse=False):

        self.project = os.path.dirname(__file__)
        self.dirname = 'data'

        self.train_data = self.load_data(dataset, 'train.txt', reverse=reverse)
        self.valid_data = self.load_data(dataset, 'valid.txt', reverse=reverse)
        self.test_data = self.load_data(dataset, 'test.txt', reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data

        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations
                                                 if i not in self.train_relations] + [i for i in self.test_relations
                                                                                      if i not in self.train_relations]

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}

    # Use a custom OpenCV function to read the image, instead of the standard
    # TensorFlow `tf.read_file()` operation.
    def _read_py_function(self, input, label):
        # image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
        label_one_hot = np.zeros(len(self.entities))
        label_one_hot[label] = 1

        input = tf.cast(input, tf.int32)
        label_one_hot = tf.constant(label_one_hot)
        label_one_hot = tf.cast(label_one_hot, tf.int32)
        # print('label length: {}'.format(label_one_hot.shape))
        # print('label: {}'.format(label_one_hot))
        return input, label_one_hot

    # The following functions can be used to convert a value to a type compatible
    # with tf.Example.
    def _bytes_feature(slef, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(self, inputs, targets):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.

        sample = {
            'inputs': self._int64_feature(inputs),
            'targets': self._bytes_feature(targets)
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=sample))

        return example_proto.SerializeToString()

    def tf_serialize_example(self, inputs, targets):

        tf_string = tf.py_func(
            self.serialize_example,
            (inputs, targets),  # pass these args to the above function.
            tf.string)      # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.

        return tf.reshape(tf_string, ())  # The result is a scalar

    def _parse_example(self, x, y):

        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.int32)

        return x, y

    def _parse_vocabulary(self, x, lst):

        lst.append(x)

        return lst

    def _parse_function_train(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        # example = tf.train.Example()
        # print(f'serialized_example: {serialized_example}')
        # example.ParseFromString(example_proto)
        # x_1 = example.features.feature['X'].float_list.value
        # y_1 = example.features.feature['Y'].float_list.value

        # sample_description = {
        #     'X': tf.FixedLenFeature([], tf.float32),
        #     'Y': tf.FixedLenFeature([], tf.float32)
        # }

        sample_description = {
            'X': tf.FixedLenFeature([2], tf.int64, default_value=[0, 0]),
            'Y': tf.FixedLenFeature([40943], tf.float32, default_value=[0.0] * 40943),
        }

        batch = tf.parse_single_example(example_proto, sample_description)
        inputs = tf.cast(batch['X'], tf.int32)
        # targets = tf.cast(batch['Y'], tf.int32)
        targets = batch['Y']

        return inputs, targets

    def _parse_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        # example = tf.train.Example()
        # print(f'serialized_example: {serialized_example}')
        # example.ParseFromString(example_proto)
        # x_1 = example.features.feature['X'].float_list.value
        # y_1 = example.features.feature['Y'].float_list.value

        # sample_description = {
        #     'X': tf.FixedLenFeature([], tf.float32),
        #     'Y': tf.FixedLenFeature([], tf.float32)
        # }

        sample_description = {
            'X': tf.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
            'Y': tf.FixedLenFeature([40943], tf.float32, default_value=[0.0] * 40943),
        }

        batch = tf.parse_single_example(example_proto, sample_description)
        inputs = tf.cast(batch['X'], tf.int32)
        # targets = tf.cast(batch['Y'], tf.int32)
        targets = batch['Y']

        return inputs, targets

    def np_to_tfrecords(self, X, Y, file_path_prefix, verbose=True):
        """
        Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
        For supervised learning, feed training inputs to X and training labels to Y.
        For unsupervised learning, only feed training inputs to X, and feed None to Y.
        The length of the first dimensions of X and Y should be the number of samples.

        Parameters
        ----------
        X : numpy.ndarray of rank 2
            Numpy array for training inputs. Its dtype should be float32, float64, or int64.
            If X has a higher rank, it should be rshape before fed to this function.
        Y : numpy.ndarray of rank 2 or None
            Numpy array for training labels. Its dtype should be float32, float64, or int64.
            None if there is no label array.
        file_path_prefix : str
            The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
        verbose : bool
            If true, progress is reported.

        Raises
        ------
        ValueError
            If input type is not float (64 or 32) or int.

        """
        def _dtype_feature(ndarray):
            """match appropriate tf.train.Feature class with dtype of ndarray. """
            assert isinstance(ndarray, np.ndarray)
            dtype_ = ndarray.dtype
            if dtype_ == np.float64 or dtype_ == np.float32:
                return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
            elif dtype_ == np.int64:
                return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
            else:
                raise ValueError("The input should be numpy ndarray. \
                                   Instaed got {}".format(ndarray.dtype))

        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2  # If X has a higher rank,
        # it should be rshape before fed to this function.
        assert isinstance(Y, np.ndarray) or Y is None

        # load appropriate tf.train.Feature class depending on dtype
        dtype_feature_x = _dtype_feature(X)
        if Y is not None:
            assert X.shape[0] == Y.shape[0]
            assert len(Y.shape) == 2
            dtype_feature_y = _dtype_feature(Y)

        # Generate tfrecord writer
        result_tf_file = file_path_prefix + '.tfrecords'
        writer = tf.python_io.TFRecordWriter(result_tf_file)
        if verbose:
            print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

        # iterate over each sample,
        # and serialize it as ProtoBuf.
        for idx in range(X.shape[0]):
            x = X[idx]
            if Y is not None:
                y = Y[idx]

            d_feature = {}
            d_feature['X'] = dtype_feature_x(x)
            if Y is not None:
                d_feature['Y'] = dtype_feature_y(y)

            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

        if verbose:
            print("Writing {} done!".format(result_tf_file))

    def load_data(self, dataset, data_type='train.txt', reverse=False):

        path = os.path.join(self.project, self.dirname, dataset, data_type)

        with open(path, 'r') as fhand:

            data = fhand.read().strip().split('\n')
            data = [sample.split() for sample in data]

            if reverse:
                data += [[triple[2], '{}_reverse'.format(triple[1]), triple[0]] for triple in data]

        return data

    def get_entities(self, data):

        entities = sorted(
            list(set([entity[0] for entity in data] + [entity[2] for entity in data])))

        return entities

    def get_relations(self, data):

        relations = sorted(list(set([relation[1] for relation in data])))

        return relations

    def get_data_idxs(self, data, entity_idxs, relation_idxs):

        data_idxs = [(entity_idxs[data[i][0]], relation_idxs[data[i][1]],
                      entity_idxs[data[i][2]]) for i in range(len(data))]

        return data_idxs

    def get_er_vocab(self, data):

        print('Constructing er_vocab...')
        er_vocab = defaultdict(list)
        # print('data before convert to tensor: {}'.format(data))
        # data = tf.convert_to_tensor(data)
        # print('data after convert to tensor: {}'.format(data))
        # for triple in data:
        #     er_vocab[(triple[0], triple[1])].append(triple[2])
        # er_vocab = data.map(self._parse_example)
        print('data type: {}'.format(type(data)))

        # print('unstacking tensor....')
        # data = tf.unstack(data, data.shape[0].value)

        print('looping over list....')
        for triple in data:
            # print('triple[2]: {}'.format(triple[2]))
            er_vocab[(triple[0], triple[1])].append(triple[2])
        # er_vocab = tf.map_fn(_parse_vocabulary, data)
        # print(er_vocab)
        # print('er_vocab: {}'.format(er_vocab))
        print('er_vocab construction complete!')

        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):

        batch = er_vocab_pairs[idx:min(idx + 128, len(er_vocab_pairs))]

        targets = np.zeros((len(batch), len(self.entities)))

        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.

        return np.array(batch), targets

    def make_source_data(self, data_idxs, data_type='train'):

        data_idxs = np.array(data_idxs)
        # data_idxs = tf.convert_to_tensor(data_idxs)

        if data_type == 'train':
            er_vocab = self.get_er_vocab(data_idxs)
            er_vocab_pairs = list(er_vocab.keys())

            # inputs = er_vocab_pairs if training else data_idxs
            inputs = er_vocab_pairs
            inputs = np.array(inputs)

            # inputs = data_idxs[:, [0, 1]]
            # inputs = tf.slice(data_idxs, [0, 0], [data_idxs.shape[0].value, 2])
            # print('inputs: {}'.format(inputs))
            # inputs = tf.convert_to_tensor(inputs)

            # set all e2 relations for e1, r pair to true
            targets = np.zeros((len(inputs), len(self.entities)))
            # labels = data_idxs[:, 2]
            # labels = tf.slice(data_idxs, [0, 2], [data_idxs.shape[0].value, 1])

            for idx, pair in enumerate(er_vocab_pairs):
                targets[idx, er_vocab[pair]] = 1.
        else:
            inputs = data_idxs
            inputs = np.array(inputs)

            targets = np.zeros((len(inputs), len(self.entities)))

            for idx, triple in enumerate(inputs):
                targets[idx, triple[2]] = 1.

            # labels = tf.convert_to_tensor(labels)

        # Assume that each row of `features` corresponds to the same row as `labels`.
        assert inputs.shape[0] == targets.shape[0]

        print('inputs shape: {}'.format(inputs.shape))
        print('targets shape: {}'.format(targets.shape))

        # with open('val_dataset.npz', 'wb') as training_set:
        #     np.savez_compressed(training_set, a=inputs, b=targets)
        #
        # print('writing to validation file complete!')

        print('writing to {} file...'.format(data_type))

        # self.np_to_tfrecords(inputs, None, 'test_record_file')
        self.np_to_tfrecords(inputs, targets, '{}_dataset'.format(data_type))

        print('writing to {} file complete!'.format(data_type))

        # for serialized_example in tf.python_io.tf_record_iterator('test_record_file.tfrecords'):
        #     example = tf.train.Example()
        #     example.ParseFromString(serialized_example)
        #     x_1 = np.array(example.features.feature['X'].float_list.value)
        #     y_1 = np.array(example.features.feature['Y'].float_list.value)
        #     print(f'sample: input: {x_1}, target: {y_1}')
        #     break

        # filenames = ['test_record_file.tfrecords']
        # raw_dataset = tf.data.TFRecordDataset(filenames)
        # raw_dataset = tf.python_io.tf_record_iterator('test_record_file.tfrecords')

        print('reading from {} file...'.format(data_type))

        raw_dataset = tf.data.TFRecordDataset('{}_dataset.tfrecords'.format(data_type))
        print('raw_dataset: {}'.format(raw_dataset))

        print('reading from {} file complete!'.format(data_type))

        parsed_dataset = raw_dataset.map(self._parse_function)
        print('parsed_dataset: {}'.format(parsed_dataset))

    def get_inputs_and_targets(self, training=False):

        if training:

            print('reading from training file...')

            # with np.load('train_dataset1.npz') as data:
            #     dataset_files = data
            #     print('dataset files: {}'.format(dataset_files.files))
            #     inputs = dataset_files['a']
            #     targets = dataset_files['b']

            raw_dataset = tf.data.TFRecordDataset('train_dataset.tfrecords')

            # raw_dataset = tf.data.TFRecordDataset(
            #     'gs://epoch-staging-bucket/hyppernetwork-factorisation/data/train_dataset.tfrecords')

            print('reading from training file complete!')

            parsed_dataset = raw_dataset.apply(
                tf.contrib.data.map_and_batch(
                    self._parse_function_train,
                    batch_size=128,
                    num_parallel_batches=None,
                    drop_remainder=True))

        else:
            # data_idxs = tf.convert_to_tensor(data_idxs)
            # print('reading from memory...')
            # inputs = tf.slice(data_idxs, [0, 0], [data_idxs.shape[0].value, 2])
            # labels = tf.slice(data_idxs, [0, 2], [data_idxs.shape[0].value, 1])
            # print('reading from memory complete!')

            print('reading from validation file...')

            # with np.load('val_dataset.npz') as data:
            #     dataset_files = data
            #     print('dataset files: {}'.format(dataset_files.files))
            #     inputs = dataset_files['a']
            #     targets = dataset_files['b']

            raw_dataset = tf.data.TFRecordDataset('val_dataset.tfrecords')

            # raw_dataset = tf.data.TFRecordDataset(
            #     'gs://epoch-staging-bucket/hyppernetwork-factorisation/data/val_dataset.tfrecords')

            print('reading from validation file complete!')

            parsed_dataset = raw_dataset.apply(
                tf.contrib.data.map_and_batch(
                    self._parse_function,
                    batch_size=128,
                    num_parallel_batches=None,
                    drop_remainder=True))

        # dataset = parsed_dataset.prefetch(tf.contrib.data.AUTOTUNE)

        # print('retrieving samples...')
        # for parsed_record in parsed_dataset.take(10):
        #     print(repr(parsed_record))

        # parsed_dataset = parsed_dataset.map(self._parse_example)

        # print('wrapping in tf.data object...')
        # data = tf.data.Dataset.from_tensor_slices((inputs, targets))
        # print('tf.data object created!')

        # data = data.map(lambda input, label: tuple(tf.py_func(
        #     self._read_py_function, [input, label], [tf.int32, tf.int32])))``

        # print('parsing to lower precision: tf.int32')
        # data = data.map(self._parse_example)
        # serialized_dataset = data.map(self.tf_serialize_example)

        return parsed_dataset
