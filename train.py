# std
from collections import defaultdict

# 3rd Party
import numpy as np
import tensorflow as tf

# internal
from load_data import Data
from model import HyperER

try:
    tf.enable_eager_execution()
    print('Running in Eager mode.')
except ValueError:
    print('Already running in Eager mode')


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

    @staticmethod
    def get_data_idxs(data, entity_idxs, relation_idxs):

        data_idxs = [(entity_idxs[data[i][0]], relation_idxs[data[i][1]],
                      entity_idxs[data[i][2]]) for i in range(len(data))]

        return data_idxs

    @staticmethod
    def get_er_vocab(data):

        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])

        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):

        batch = er_vocab_pairs[idx:min(idx + self.batch_size, len(er_vocab_pairs))]

        # set all e2 relations for e1, r pair to true
        targets = np.zeros((len(batch), len(self.data.entities)))

        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.

        targets = tf.convert_to_tensor(targets)

        return np.array(batch), targets

    def loss(self, e1_idx, r_idx, targets):

        predictions = self.model(e1_idx, r_idx, training=True)
        loss = tf.keras.backend.binary_crossentropy(
            tf.cast(targets, tf.double), tf.cast(predictions, tf.double))
        # loss = binary_crossentropy(targets, predictions, from_logits=False)
        # loss = tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(targets, 1), predictions)
        # loss = tf.losses.softmax_cross_entropy(targets, predictions)
        cost = tf.reduce_mean(loss)

        # loss = tf.cast(targets, tf.double) * tf.cast(tf.log(predictions), tf.double)
        # cost = -tf.reduce_sum(loss)

        return cost

    def train_and_eval(self):
        # Prepare train input and targets
        # Prepare training data
        train_data_idxs = self.get_data_idxs(
            self.data.train_data, self.model.entity_idxs, self.model.relation_idxs)

        er_vocab = self.get_er_vocab(train_data_idxs)

        view_key = list(er_vocab.keys())[0]
        print('sample enitity-relation: {}'.format(view_key))

        er_vocab_pairs = list(er_vocab.keys())
        print('er_vocab_pairs: {}'.format(len(er_vocab_pairs)))

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        losses = []

        # Training loop
        for epoch in range(self.num_epoch):

            print(f'epoch: {epoch + 1}')

            iteration = 0

            for j in range(0, len(er_vocab_pairs), self.batch_size):
                # x, y
                train_batch, train_targets = self.get_batch(er_vocab, er_vocab_pairs, j)

                # Entitty and relation training ids
                e1_idx = train_batch[:, 0]
                r_idx = train_batch[:, 1]

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * train_targets) + \
                        (1.0 / train_targets.shape[1].value)
                else:
                    targets = train_targets

                optimizer.minimize(lambda: self.loss(e1_idx, r_idx, targets))

                if j % (self.batch_size * 10) == 0:
                    print(f'iteration: {iteration + 1}')

                    cost = self.loss(e1_idx, r_idx, targets)
                    print(f'cost: {cost}')
                    # print(f'predictions: {model(r, e1, e2)}')

                iteration += 1

            losses.append(cost)
            print(f'mean cost: {np.mean(losses)}')
            print()

            # Validate model
            self.evaluate()

    def evaluate(self):

        hits = []
        ranks = []

        for i in range(10):
            hits.append([])

        valid_data_idxs = self.get_data_idxs(
            self.data.valid_data, self.model.entity_idxs, self.model.relation_idxs)
        er_vocab = self.get_er_vocab(valid_data_idxs)

        print(f'Number of data points: {len(valid_data_idxs)}')

        for i in range(0, len(valid_data_idxs), self.batch_size):

            valid_batch, _ = self.get_batch(er_vocab, valid_data_idxs, i)

            e1_idx = valid_batch[:, 0]
            r_idx = valid_batch[:, 1]
            e2_idx = valid_batch[:, 2]

            predictions = self.model(e1_idx, r_idx)  # model.forward(e1_idx, r_idx)

            sort_idxs = tf.argsort(predictions, axis=1, direction='DESCENDING')

            for j in range(valid_batch.shape[0]):

                rank = np.where(np.array(sort_idxs[j]) == e2_idx[j])[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):

                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        print()


if __name__ == '__main__':

    # Load data
    data = Data(dataset='WN18', reverse=True)

    entities = data.entities
    relations = data.relations

    # Intialise model
    hypER = HyperER(entities, relations)

    # intialise build
    trainer = Train(hypER, data, num_epoch=10)
    trainer.train_and_eval()
