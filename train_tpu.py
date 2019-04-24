# std
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from datetime import datetime

# 3rd party
import tensorflow as tf

# internal
from load_data import Data
from model import HyperER


# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default='luyolo-nqobile',
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "gs://epoch-staging-bucket/hyppernetwork-factorisation/data",
                       "Path to directory containing the HypER dataset")
tf.flags.DEFINE_string(
    "model_dir", 'gs://epoch-staging-bucket/hyppernetwork-factorisation/output', "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1337 * 20, "Total number of training steps.")  # 13370
tf.flags.DEFINE_integer("eval_steps", 78,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS


def train_input_fn(params):
    """train_input_fn defines the input pipeline used for training."""

    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    data = Data(dataset='WN18', reverse=True)

    train_data = data.get_inputs_and_targets(training=True)

    ds = train_data.shuffle(buffer_size=1000).repeat()

    return ds


def eval_input_fn(params):

    data = Data(dataset='WN18', reverse=True)

    validation_data = data.get_inputs_and_targets()

    ds = validation_data

    return ds


def predict_input_fn(params):

    batch_size = params["batch_size"]

    data = Data(dataset='WN18', reverse=True)

    test_data = data.get_inputs_and_targets()

    # Take out top 10 samples from test data to make the predictions.
    ds = test_data.take(10).batch(batch_size)

    return ds


def metric_fn(features, labels, logits):

    hits = []
    ranks = []

    for i in range(10):
        hits.append([])

    samples = []

    data = Data(dataset='WN18', reverse=True)

    valid_data_idxs = data.get_data_idxs(
        data.valid_data, data.entity_idxs, data.relation_idxs)

    # test_data_idxs = data.get_data_idxs(data)
    er_vocab = data.get_er_vocab(data.get_data_idxs(
        data.valid_data, data.entity_idxs, data.relation_idxs))

    for i in range(0, len(valid_data_idxs), 128):

        data_batch, _ = data.get_batch(er_vocab, valid_data_idxs, i)

        e2_idx = data_batch[:, 2]

    for j in range(data_batch.shape[0]):

        logits_list = []

        filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
        target_value = logits[j][e2_idx[j]]

        logits_unstacked = tf.unstack(logits[j])

        for it in range(len(logits_unstacked)):
            logits_list.append(logits_unstacked[it])

        for k in range(len(filt)):
            result = tf.cond(tf.equal(e2_idx[j], filt[k]), lambda: target_value, lambda: 0.0)
            logits_list[filt[k]] = result

        samples.append(logits_list)

    e2_idx = labels
    logits = tf.stack(samples)

    sort_idxs = tf.argsort(logits, axis=1, direction='DESCENDING')

    for j in range(logits.shape[0]):

        rank = tf.where(tf.equal(sort_idxs[j], e2_idx[j]))
        ranks.append(rank + 1)

        for hits_level in range(10):

            result = tf.cond(tf.squeeze(rank) <= hits_level, lambda: 1.0, lambda: 0.0)
            hits[hits_level].append(result)

    accuracy = tf.metrics.mean(hits[9])

    return {"accuracy": accuracy}


def model_fn(features, labels, mode, params):
    """model_fn constructs the ML model used to predict handwritten digits."""

    del params

    targets = labels
    targets = ((1.0 - 0.1) * targets) + (1.0 / targets.shape[1].value)  # label smoothing

    print('inputs: {}'.format(features))

    print('features shape: {}'.format(features.shape))
    e1_idx = tf.slice(features, [0, 0], [features.shape[0].value, 1])
    print('e1_idx: {}'.format(e1_idx))
    r_idx = tf.slice(features, [0, 1], [features.shape[0].value, 1])
    labels = tf.slice(features, [0, 2], [features.shape[0].value, 1])

    data = Data(dataset='WN18', reverse=False)
    model = HyperER(len(data.entities), len(data.relations))

    if mode == tf.estimator.ModeKeys.PREDICT:

        logits = model(e1_idx, r_idx, training=False)
        predictions = {
            'class_ids': tf.argmax(logits, axis=1),
            'probabilities': tf.sigmoid(logits),
        }

        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

    logits = model(e1_idx, r_idx, training=(mode == tf.estimator.ModeKeys.TRAIN))
    predictions = tf.sigmoid(logits)

    print('number of inputs: {}'.format(features.shape[0].value))

    loss = tf.keras.losses.binary_crossentropy(
        tf.cast(targets, tf.float32), tf.cast(predictions, tf.float32))

    loss = tf.reduce_mean(loss)
    global_step = tf.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:

        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps=1337,
            decay_rate=0.99,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss, eval_metrics=(metric_fn, [features, labels, logits]))


if __name__ == '__main__':

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size,
        params={"data_dir": FLAGS.data_dir},
        config=run_config)

    # TPU Estimator.train *requires* a max_steps argument.
    print('########################## RUNNING TRAINING ############################')
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
    print('########################## COMPLETED TRAINING ##########################')

    # TPUEstimator.evaluate *requires* a steps argument.
    # Note that the number of examples used during evaluation is
    # --eval_steps * --batch_size.
    # So if you change --batch_size then change --eval_steps too.
    if FLAGS.eval_steps:
        print('########################## RUNNING VALIDATION ############################')
        evaluations = estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)

        template = ('Accuracy is {}. cost: {}')

        cost = evaluations['loss']
        accuracy = evaluations['accuracy']

        print(template.format(100 * accuracy, cost))
        print(datetime.now())
        print('########################## COMPLETED VALIDATION ##########################')

    # # Run prediction on top few samples of test data.
    # if FLAGS.enable_predict:
    #     print('########################## RUNNING TESTING ############################')
    #     predictions = estimator.predict(input_fn=predict_input_fn)
    #     print('########################## COMPLETED TESTING ##########################')
    #
    # for pred_dict in predictions:
    #     template = ('Prediction is "{}" ({:.1f}%).')
    #
    #     class_id = pred_dict['class_ids']
    #     probability = pred_dict['probabilities'][class_id]
    #
    #     print(template.format(class_id, 100 * probability))
