# std
import sys
import logging

# 3rd Party
import numpy as np
import tensorflow as tf

# internal
from load_data import Data
from model import HyperER

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('train.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

try:
    tf.enable_eager_execution()
    logger.info('Running in Eager mode.')
except ValueError:
    logger.info('Already running in Eager mode')


class Train:

    def __init__(self,
                 model,
                 data,
                 learning_rate=0.001,
                 num_epoch=10,
                 batch_size=128,
                 decay_rate=0.,
                 label_smoothing=0.):

        self.model = model
        self.data = data
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing

    def loss(self, e1_idx, r_idx, targets):

        logits = self.model(e1_idx, r_idx, training=True)
        # print(f'logits: {logits}')
        predictions = tf.sigmoid(logits)
        loss = tf.keras.backend.binary_crossentropy(
            tf.cast(targets, tf.float32), tf.cast(predictions, tf.float32))
        # loss = binary_crossentropy(targets, predictions, from_logits=False)
        # loss = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions)
        # loss = tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(targets, 1), predictions)

        # loss = tf.losses.softmax_cross_entropy(targets, predictions)
        cost = tf.reduce_mean(loss)

        # loss = tf.cast(targets, tf.double) * tf.cast(tf.log(predictions), tf.double)
        # cost = -tf.reduce_sum(loss)

        return cost

    def train_and_eval(self):

        learning_rate = self.learning_rate

        optimizer = tf.train.AdamOptimizer(learning_rate)

        losses = []

        # Training loop
        for epoch in range(self.num_epoch):

            logger.info(f'epoch: {epoch + 1}')

            iteration = 0

            logger.info('loading data set...')
            train_data = self.data.get_inputs_and_targets(training=True)
            logger.info('train_data: {}'.format(train_data))
            logger.info('loaded dataset!')

            for train_inputs, train_targets in train_data:

                # print(f'inputs_train: {train_inputs}')

                # print('getting er_vocab....')
                # er_vocab = self.data.get_er_vocab(inputs_train)
                # print('er vocab type: {}'.format(type(er_vocab)))
                # er_vocab_pairs = list(er_vocab.keys())

                # inputs_train = np.array(er_vocab_pairs)

                # Entitty and relation training ids
                # e1_idx = inputs_train[:, 0]
                e1_idx = tf.slice(train_inputs, [0, 0], [train_inputs.shape[0].value, 1])
                # print('e1_idx: {}'.format(e1_idx))
                # r_idx = inputs_train[:, 1]
                r_idx = tf.slice(train_inputs, [0, 1], [train_inputs.shape[0].value, 1])
                # print('targets_train: {}'.format(targets_train))

                # targets_train = np.zeros((len(inputs_train), len(self.data.entities)))
                # for idx, pair in enumerate(er_vocab_pairs):
                #     print('index: {}'.format(er_vocab[pair]))
                #     targets_train[idx, er_vocab[pair]] = 1.

                # if self.label_smoothing:
                #     targets_train = ((1.0 - self.label_smoothing) * targets_train) + \
                #         (1.0 / targets_train.shape[1].value)

                extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_ops):
                    optimizer.minimize(lambda: self.loss(e1_idx, r_idx, train_targets),
                                       global_step=tf.train.get_or_create_global_step())

                if iteration % 10 == 0:
                    logger.info('ITERATION: {}'.format(iteration + 1))

                    cost = self.loss(e1_idx, r_idx, train_targets)
                    logger.info('cost: {}'.format(cost))

                iteration += 1

            losses.append(cost)
            logger.info('mean cost: {}'.format(np.mean(losses)))
            logger.info(' ')

            # Validate model
            self.evaluate()

    def evaluate(self):

        hits = []
        ranks = []

        for i in range(10):
            hits.append([])

        validation_data = self.data.get_inputs_and_targets()

        for val_inputs, _ in validation_data:

            # inputs_validation = np.array(inputs_validation)

            # e1_idx = inputs_validation[:, 0]
            # e1_idx = tf.slice(inputs_validation, [0, 0], [inputs_validation.shape[0].value, 1])
            e1_idx = tf.slice(val_inputs, [0, 0], [val_inputs.shape[0].value, 1])
            # e1_idx = tf.cast(e1_idx, tf.int32)
            # r_idx = inputs_validation[:, 1]
            # r_idx = tf.slice(inputs_validation, [0, 1], [inputs_validation.shape[0].value, 1])
            r_idx = tf.slice(val_inputs, [0, 1], [val_inputs.shape[0].value, 1])
            # e2_idx = inputs_validation[:, 2]
            # e2_idx = tf.slice(inputs_validation, [0, 2], [inputs_validation.shape[0].value, 1])
            e2_idx = tf.slice(val_inputs, [0, 2], [val_inputs.shape[0].value, 1])

            logits = self.model(e1_idx, r_idx)

            # print(f'logits: {logits.shape}')
            # e2_idx = tf.convert_to_tensor(e2_idx)
            sort_idxs = tf.argsort(logits, axis=1, direction='DESCENDING')

            # print('e2_idx[0]: {}'.format(e2_idx[0]))
            # print('tf.where: {}'.format(tf.where(tf.equal(sort_idxs[0], e2_idx[0]))))

            for j in range(val_inputs.shape[0]):

                rank = tf.where(tf.equal(sort_idxs[j], e2_idx[j]))
                ranks.append(rank + 1)

                for hits_level in range(10):

                    # if rank <= hits_level:
                    #     hits[hits_level].append(1.0)
                    # else:
                    #     hits[hits_level].append(0.0)

                    # print('rank: {}'.format(tf.squeeze(rank)))
                    # print('hits_level: {}'.format(hits_level))
                    result = tf.cond(tf.squeeze(rank) <= hits_level, lambda: 1.0, lambda: 0.0)
                    hits[hits_level].append(result)

        logger.info('Hits @10: {0}'.format(tf.reduce_mean(hits[9])))
        logger.info('Hits @3: {0}'.format(tf.reduce_mean(hits[2])))
        logger.info('Hits @1: {0}'.format(tf.reduce_mean(hits[0])))
        logger.info('Mean rank: {0}'.format(tf.reduce_mean(ranks)))
        logger.info('Mean reciprocal rank: {0}'.format(tf.reduce_mean(
            tf.math.reciprocal(tf.cast(ranks, tf.float32)))))
        logger.info(' ')

    def test(self):

        hits = []
        ranks = []

        for i in range(10):
            hits.append([])

        test_data = self.data.get_inputs_and_targets(test_data_idxs)

        for inputs_test, _ in test_data.batch(self.batch_size):

            inputs_test = np.array(inputs_test)

            e1_idx = inputs_test[:, 0]
            r_idx = inputs_test[:, 1]
            e2_idx = inputs_test[:, 2]

            logits = self.model(e1_idx, r_idx)

            sort_idxs = tf.argsort(logits, axis=1, direction='DESCENDING')

            for j in range(inputs_test.shape[0]):

                rank = np.where(np.array(sort_idxs[j]) == e2_idx[j])[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):

                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info('Hits @10: {0}'.format(np.mean(hits[9])))
        logger.info('Hits @3: {0}'.format(np.mean(hits[2])))
        logger.info('Hits @1: {0}'.format(np.mean(hits[0])))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))


if __name__ == '__main__':

    # Load data
    data = Data(dataset='WN18', reverse=True)

    # Intialise model
    hypER = HyperER(len(data.entities), len(data.relations))

    # intialise build
    trainer = Train(hypER, data, num_epoch=100)
    trainer.train_and_eval()
    # trainer.evaluate()
    # trainer.test()
