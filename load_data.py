import os


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

    def load_data(self, dataset, data_type='train.txt', reverse=False):

        path = os.path.join(self.project, self.dirname, dataset, data_type)

        with open(path, 'r') as fhand:

            data = fhand.read().strip().split('\n')
            data = [sample.split() for sample in data]

            if reverse:
                data += [[triple[2], f'{triple[1]}_reverse', triple[0]] for triple in data]

        return data

    def get_entities(self, data):

        entities = sorted(
            list(set([entity[0] for entity in data] + [entity[2] for entity in data])))

        return entities

    def get_relations(self, data):

        relations = sorted(list(set([relation[1] for relation in data])))

        return relations

    def get_path(self, filename):

        dirname = os.path.dirname(__file__)
        filename = filename
        path = os.path.join(dirname, filename)

        return path
