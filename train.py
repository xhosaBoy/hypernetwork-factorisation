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
                 num_epoch=100,
                 batch_size=128,
                 decay_rate=0.99,
                 label_smoothing=0.1):

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

    def accuracy(self, e1_idx, r_idx, targets):

        # labels = torch.max(targets, 1)[1]
        # classes = torch.max(logits, 1)[1]

        # accuracy = torch.eq(classes, labels)
        # accuracy = torch.sum(accuracy).float() / targets.size(0)

        logits = self.model(e1_idx, r_idx, training=True)

        labels = tf.argmax(targets, 1)
        classes = tf.argmax(logits, 1)

        equality = tf.equal(tf.cast(classes, tf.int32), tf.cast(labels, tf.int32))
        acc = tf.reduce_mean(tf.cast(equality, tf.float32))

        # output = (predictions > 0.5).float()
        # correct = (output == targets).float().sum()
        # accuracy = correct / output.shape[0]

        return acc

    def train_and_eval(self):

        # learning_rate = self.learning_rate
        # global_step = tf.Variable(0, trainable=False)
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            global_step,
            decay_steps=1337,
            decay_rate=self.decay_rate,
            staircase=True)

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

                # ogger.info(f'itration: {iteration + 1}')

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

                if self.label_smoothing:
                    train_targets = ((1.0 - self.label_smoothing) * train_targets) + \
                        (1.0 / train_targets.shape[1].value)
                # print('train_targets size: {}'.format(train_targets.shape[1].value))

                extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                cost = self.loss(e1_idx, r_idx, train_targets)

                with tf.control_dependencies(extra_ops):
                    optimizer.minimize(lambda: self.loss(e1_idx, r_idx, train_targets),
                                       global_step=global_step)

                if iteration % 10 == 0:
                    logger.info('ITERATION: {}'.format(iteration + 1))

                    cost = self.loss(e1_idx, r_idx, train_targets)
                    logger.info('cost: {}'.format(cost))

                    accuracy = self.accuracy(e1_idx, r_idx, train_targets)
                    logger.info('accuracy: {}'.format(accuracy))

                    logger.info(f'learning_rate: {learning_rate()}')
                    logger.info(f'global_step: {global_step}')

                iteration += 1

            losses.append(cost)
            logger.info('mean cost: {}'.format(np.mean(losses)))

            # Validate model
            self.evaluate()

    def evaluate(self):

        hits = []
        ranks = []

        for i in range(10):
            hits.append([])

        # validation_data = self.data.get_inputs_and_targets()
        val_data_idxs = self.data.get_data_idxs(
            self.data.valid_data, self.data.entity_idxs, self.data.relation_idxs)
        er_vocab = self.data.get_er_vocab(val_data_idxs)

        print("Number of data points: %d" % len(val_data_idxs))

        iteration = 0

        # for val_inputs, _ in validation_data:
        #
        #     # if iteration % 10 == 0:
        #     logger.info('ITERATION: {}'.format(iteration + 1))
        #
        #     samples = []
        #
        #     # inputs_validation = np.array(inputs_validation)
        #
        #     # e1_idx = inputs_validation[:, 0]
        #     # e1_idx = tf.slice(inputs_validation, [0, 0], [inputs_validation.shape[0].value, 1])
        #     e1_idx = tf.slice(val_inputs, [0, 0], [val_inputs.shape[0].value, 1])
        #     # e1_idx = tf.cast(e1_idx, tf.int32)
        #     # r_idx = inputs_validation[:, 1]
        #     # r_idx = tf.slice(inputs_validation, [0, 1], [inputs_validation.shape[0].value, 1])
        #     r_idx = tf.slice(val_inputs, [0, 1], [val_inputs.shape[0].value, 1])
        #     # e2_idx = inputs_validation[:, 2]
        #     # e2_idx = tf.slice(inputs_validation, [0, 2], [inputs_validation.shape[0].value, 1])
        #     e2_idx = tf.slice(val_inputs, [0, 2], [val_inputs.shape[0].value, 1])
        #
        #     logits = self.model(e1_idx, r_idx)

        # logits = tf.unstack(logits)

        # for j in range(tensor_shape[0]):
        #     logits_list.append(logits[j])

        # logits_list = np.array(logits_list)
        # print('logits shape: {}'.format(logits.shape))

        # for j in range(val_inputs.shape[0]):
        #
        #     logits_list = []
        #
        #     filt = er_vocab[(val_inputs[j][0], val_inputs[j][1])]
        #     # print('er_vocab: {}'.format(
        #     #     (val_inputs[j][0].numpy(), val_inputs[j][1].numpy())))
        #
        #     # print('len(filt): {}'.format(len(filt)))
        #     # print('e2_idx: {}'.format(e2_idx[j][0]))
        #
        #     target_value = logits[j][e2_idx[j][0]]
        #
        #     # logits_list[j, filt] = 0.0
        #     logits_unstacked = tf.unstack(logits[j])
        #     # print('unstacked len: {}'.format(len(logits_unstacked)))
        #
        #     for it in range(len(logits_unstacked)):
        #         logits_list.append(logits_unstacked[it])
        #
        #     for k in range(len(filt)):
        #         result = tf.cond(tf.equal(e2_idx[j], filt[k]),
        #                          lambda: target_value, lambda: 0.0)
        #         logits_list[filt[k]] = result
        #         # logits_list[filt[k]] = 0.0

        for i in range(0, len(val_data_idxs), self.batch_size):

            data_batch, _ = self.data.get_batch(er_vocab, val_data_idxs, i)

            e1_idx = data_batch[:, 0]
            r_idx = data_batch[:, 1]
            e2_idx = data_batch[:, 2]

            logits = self.model(e1_idx, r_idx)
            logits = np.array(logits)

            for j in range(data_batch.shape[0]):

                # filt = er_vocab[(val_inputs[j][0].numpy(), val_inputs[j][1].numpy())]
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                # print('filt: {}'.format(filt))
                target_value = logits[j, e2_idx[j]]
                # print('target_value: {}'.format(target_value))
                logits[j, filt] = 0.0
                logits[j, e2_idx[j]] = target_value

            # logits_list[e2_idx[j][0]] = target_value
            # print('gather: {}'.format(tf.gather(tf.unstack(e2_idx), j)))
            # index = tf.gather(tf.unstack(e2_idx), j)[0]
            # print('index: {}'.format(index))
            # logits_list[index] = target_value

            # print('logits list len: {}'.format(len(logits_list)))
            # samples.append(logits_list)

            # print(f'logits: {logits.shape}')
            # e2_idx = tf.convert_to_tensor(e2_idx)
            # sort_idxs = tf.argsort(logits, axis=1, direction='DESCENDING')
            # logits = tf.stack(samples)

            logits = tf.convert_to_tensor(logits)
            sort_idxs = tf.argsort(logits, axis=1, direction='DESCENDING')
            sort_idxs = sort_idxs.cpu().numpy()

            # print('e2_idx[0]: {}'.format(e2_idx[0]))
            # print('tf.where: {}'.format(tf.where(tf.equal(sort_idxs[0], e2_idx[0]))))

            for j in range(data_batch.shape[0]):

                rank = np.where(sort_idxs[j] == e2_idx[j])[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):

                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

                    # print('rank: {}'.format(tf.squeeze(rank)))
                    # print('hits_level: {}'.format(hits_level))
                    # result = tf.cond(tf.squeeze(rank) <= hits_level, lambda: 1.0, lambda: 0.0)
                    # hits[hits_level].append(result)

            iteration += 1

        logger.info('Hits @10: {0}'.format(np.mean(hits[9])))
        logger.info('Hits @3: {0}'.format(np.mean(hits[2])))
        logger.info('Hits @1: {0}'.format(np.mean(hits[0])))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

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
