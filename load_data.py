# std
import os
import json
import pickle
from collections import defaultdict

# 3rd Party
import numpy as np
import h5py
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

    def _parse_example(self, x, y):

        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.int32)

        return x, y

    def _parse_vocabulary(self, x, lst):

        lst.append(x)

        return lst

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
        print('er_vocab construction complete...!')

        return er_vocab

    def get_inputs_and_targets(self, data_idxs, training=False):

        # er_vocab = self.get_er_vocab(data_idxs)
        # er_vocab_pairs = list(er_vocab.keys())

        # inputs = er_vocab_pairs if training else data_idxs
        data_idxs = tf.convert_to_tensor(data_idxs)
        # inputs = data_idxs[:, [0, 1]]
        inputs = tf.slice(data_idxs, [0, 0], [data_idxs.shape[0].value, 2])
        # print('inputs: {}'.format(inputs))
        # inputs = tf.convert_to_tensor(inputs)

        # inputs = inputs[:10000]
        # targets = targets[:10000]

        # set all e2 relations for e1, r pair to true
        # labels = np.zeros((len(inputs), len(self.entities)))
        # labels = data_idxs[:, 2]
        labels = tf.slice(data_idxs, [0, 2], [data_idxs.shape[0].value, 1])

        # for idx, pair in enumerate(er_vocab_pairs):
        #     labels[idx] = er_vocab[pair]

        # labels = tf.convert_to_tensor(labels)

        # print('writing to file...')

        # PICKLE Format
        # pickle.dump(targets_truncated, output)
        # # pickle.dump(data1, output)
        # output.close()
        #
        # pkl_file = open('data.pkl', 'rb')
        #
        # data1 = pickle.load(pkl_file)
        # pprint.pprint(data1)
        #
        # pkl_file.close()

        # with open('data.pkl', 'wb') as output_file:
        #     pickle.dump(targets, output_file)

        # NUMPY Format
        # output = open('data.npy', 'wb')
        # a = targets
        # np.save(output, a)
        # output.close()
        #
        # input = open('data.npy', 'rb')
        # targets = np.load(input)
        # aa = z['a']
        # input.close()

        # with open('data.npz', 'wb') as training_set:
        #     np.savez_compressed(training_set, a=labels)

        # with np.load('data.npz') as data:
        #     label_file = data
        #     print('label_file files: {}'.format(label_file.files))
        #     labels = label_file['a']

        # HDF5 Format
        # hdf5_store = h5py.File("./cache.hdf5", "w")
        # targets = hdf5_store.create_dataset(
        #     "targets", (len(inputs), len(self.entities)), compression="gzip")
        #
        # for idx, pair in enumerate(inputs):
        #     # print('er_vocab[pair] type: {}'.format(type(tuple(er_vocab[pair]))))
        #     for e in er_vocab[pair]:
        #         targets[idx, e] = 1.

        # hdf5_store = h5py.File("./cache.hdf5", "r")
        #
        # labels = hdf5_store["targets"]
        # print('writing to file complete!')

        # print('labels: {}'.format(labels))
        print('inputs shape: {}'.format(inputs.shape))
        print('labels shape: {}'.format(labels.shape))

        # Assume that each row of `features` corresponds to the same row as `labels`.
        assert inputs.shape[0] == labels.shape[0]

        # inputs_placeholder = tf.placeholder(inputs.dtype, inputs.shape)
        # labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

        # print('writing to file complete!')

        print('wrapping in tf.data object...')
        # data = tf.data.Dataset.from_tensor_slices((inputs_placeholder, labels_placeholder))
        data = tf.data.Dataset.from_tensor_slices((inputs, labels))

        # data = data.map(lambda input, label: tuple(tf.py_func(
        #     self._read_py_function, [input, label], [tf.int32, tf.int32])))

        # print('parsing to lower precision: tf.int32')
        data = data.map(self._parse_example)

        return data

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
