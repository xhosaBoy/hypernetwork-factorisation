import tensorflow as tf


class HyperER(tf.keras.Model):

    def __init__(self, num_entities, num_relations):

        super(HyperER, self).__init__()

        self.entity_dim = 200
        self.relation_dim = 200
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.in_channels = 1
        self.out_channels = 32
        self.kernal_h = 1
        self.kernal_w = 9

        self.dense1_size_out = self.in_channels * self.out_channels * self.kernal_h * self.kernal_w
        self.dense2_size_in = (1 - self.kernal_h + 1) * (self.entity_dim -
                                                         self.kernal_w + 1) * self.out_channels

        # self.inp_drop = 0.2
        # self.feature_map_drop = 0.2
        # self.hidden_drop = 0.3

        self.weights_dense1 = tf.Variable(lambda: tf.glorot_normal_initializer()(
            [self.relation_dim, self.dense1_size_out]))
        self.bias_dense1 = tf.Variable(
            lambda: tf.glorot_normal_initializer()([self.dense1_size_out]))

        self.weights_dense2 = tf.Variable(lambda:
                                          tf.glorot_normal_initializer()([self.dense2_size_in, self.entity_dim]))
        self.bias_dense2 = tf.Variable(lambda: tf.glorot_normal_initializer()([self.entity_dim]))

        # self.bias_logits = tf.Variable(
        #     lambda: tf.glorot_normal_initializer()([128, self.num_entities]))
        self.bias_logits = tf.Variable(
            lambda: tf.glorot_normal_initializer()([self.num_entities]))

        # Generate random embedding representaitons for  and relations
        self.embedding_matrix_entities = tf.Variable(lambda:
                                                     tf.glorot_normal_initializer()([self.num_entities, self.entity_dim]))
        self.embedding_matrix_relations = tf.Variable(lambda:
                                                      tf.glorot_normal_initializer()([self.num_relations, self.relation_dim]))

        self.bn0 = tf.keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-05)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-05)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05)

        self.inp_drop = tf.keras.layers.Dropout(0.2)
        self.feature_map_drop = tf.keras.layers.SpatialDropout2D(0.2)
        self.hidden_drop = tf.keras.layers.Dropout(0.3)

        # self.dense1 = tf.keras.layers.Dense(self.dense1_size_out)
        # self.dense2 = tf.keras.layers.Dense(self.entity_dim)

        self.add = tf.keras.layers.Add()

    def conv1(self, x, k):
        conv_layer = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='VALID')

        return conv_layer

    def dense1(self, x):
        dense_layer = tf.matmul(x, self.weights_dense1)
        dense_layer += self.bias_dense1

        return dense_layer

    def dense2(self, x):
        dense_layer = tf.matmul(x, self.weights_dense2)
        dense_layer += self.bias_dense2
        # dense_layer = tf.nn.relu(dense_layer)

        return dense_layer

    # def dropout(self, x, training=False, keep_prob=0.):
    #     keep_prob = keep_prob if training else 0.
    #     dropout_layer = tf.nn.dropout(x, keep_prob)
    #
    #     return dropout_layer

    def call(self, e1_idx, r_idx, training=False):

        # Get embedding weights for forward pass
        e1 = tf.nn.embedding_lookup(self.embedding_matrix_entities, [e1_idx])
        r = tf.nn.embedding_lookup(self.embedding_matrix_relations, [r_idx])
        e2 = tf.nn.embedding_lookup(self.embedding_matrix_entities, [
                                    id for id in range(self.num_entities)])

        # Compute hyper relational filters
        r = tf.reshape(r, [-1, 200])
        # out (MB, self.in_channels * self.out_channels * self.kernal_h * self.kernal_w)
        k = self.dense1(r)

        # Depthwise Convolution
        # k has shape (MB, fh, fw, in_channels, out_channels)
        k = tf.reshape(k, [r.shape[0].value, self.kernal_h, self.kernal_w,
                           self.in_channels, self.out_channels])
        # k has shape (fh, fw, MB, in_channels, out_channels)
        k = tf.transpose(k, perm=[1, 2, 0, 3, 4])
        # k has shape (fh, fw, in_channels * MB, out_channels)
        k = tf.reshape(k, [self.kernal_h, self.kernal_w, self.in_channels *
                           r.shape[0].value, self.out_channels])

        # Depthwise Prepare subject entity for 2D convolution
        # x has shape (MB, H, W, in_channels)
        x = tf.reshape(e1, [-1, 1, self.entity_dim, self.in_channels])

        # x = self.bn0(x, training=training)
        # if training:
        #     x = self.inp_drop(x)
        x = self.inp_drop(x, training=training)

        # Depthwise Convolution
        x = tf.transpose(x, perm=[1, 2, 0, 3])  # x has shape (H, W, MB, in_channels)
        # x has shape (1, H, W, MB * in_channels)
        x = tf.reshape(x, [1, 1, self.entity_dim, e1.shape[1].value * self.in_channels])

        # Take Convolution
        x = self.conv1(x, k)  # x has shape (1, fd, convultion(x, f), MB * out_channels))

        # Depthwise Convolution
        # x has shape (H - fh + 1, W - fw + 1, MB, in_channels, out_channels)
        x = tf.reshape(x, [1 - self.kernal_h + 1, self.entity_dim - self.kernal_w +
                           1, e1.shape[1].value, self.in_channels, self.out_channels])

        # x has shape (MB, H - fh + 1, W - fw + 1, in_channels, out_channels)
        x = tf.transpose(x, [2, 0, 1, 3, 4])
        x = tf.reduce_sum(x, axis=3)  # x has shape (MB, H - fh + 1, W - fw + 1, out_channels)
        # x = tf.transpose(x, [0, 3, 1, 2])  # x has shape (MB, out_channels, H - fh + 1, W - fw + 1)
        x = tf.transpose(x, [0, 3, 1, 2])  # x has shape (MB, out_channels, H - fh + 1, W - fw + 1)

        # x = self.bn1(x, training=training)
        # if training:
        #     x = self.feature_map_drop(x)
        x = self.feature_map_drop(x, training=training)

        # Fully connected layer
        x = tf.reshape(x, [e1.shape[1].value, -1])  # out 128 x 6144

        x = self.dense2(x)  # out shape 128 x 200

        x = self.hidden_drop(x, training=training)
        # x = self.bn2(x, training=training)

        x = tf.nn.relu(x)

        # x = self.bn2(x, training=training)
        # if training:
        #     x = self.hidden_drop(x)
        # x = self.hidden_drop(x, training=training)

        # Link prediction logits
        logits = tf.matmul(x, tf.transpose(e2))
        # logits = self.add([logits, tf.broadcast_to(self.bias_logits, logits.shape)])
        # logits = self.add([logits, self.bias_logits])
        bias_logits = tf.broadcast_to(self.bias_logits, [logits.shape[0].value, 40943])
        logits = self.add([logits, bias_logits])

        return logits
