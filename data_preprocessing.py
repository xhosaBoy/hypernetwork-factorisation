from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# internal
from load_data import Data


if __name__ == '__main__':

    dataset = []
    # Load data
    data = Data(dataset='WN18', reverse=True)

    # Prepare train input and targets
    train_data_idxs = data.get_data_idxs(
        data.train_data, data.entity_idxs, data.relation_idxs)

    print('Number of training data points: {}'.format(len(train_data_idxs)))

    dataset.append({'name': 'train', 'data': train_data_idxs})

    valid_data_idxs = data.get_data_idxs(
        data.valid_data, data.entity_idxs, data.relation_idxs)

    print('Number of validation data points: {}'.format(len(valid_data_idxs)))

    dataset.append({'name': 'val', 'data': valid_data_idxs})

    test_data_idxs = data.get_data_idxs(
        data.test_data, data.entity_idxs, data.relation_idxs)

    print('Number of test data points: {}'.format(len(test_data_idxs)))

    dataset.append({'name': 'test', 'data': test_data_idxs})

    for data_type in dataset:
        data.make_source_data(data_type['data'], data_type['name'])
