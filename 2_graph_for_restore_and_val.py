# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
from SurvivalAnalysis import SurvivalAnalysis
import data_providers
import cox_layer
import dnn_model
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('x_data', './data/Brain_Integ_X.csv',
                    'Directory file with features-data.')
flags.DEFINE_string('y_data', './data/Brain_Integ_Y.csv',
                    'Directory file with label-values.')
flags.DEFINE_string('ckpt_dir', './ckpt_dir/',
                    'Directory for checkpoint files.')
flags.DEFINE_float('split_ratio', 0.6,
                   'Split ratio for test data.')
flags.DEFINE_float('lr_decay_rate', 0.9,
                   'Learning decaying rate.')
flags.DEFINE_float('beta', 0.01,
                   'Regularizing constant.')
flags.DEFINE_float('dropout', 0.6,
                   'Drop out ratio.')
flags.DEFINE_float('init_lr', 0.001,
                   'Initial learning rate.')
flags.DEFINE_integer('batch_size', 100,
                     'Batch size.')
flags.DEFINE_integer('n_epochs', 80,
                     'Number of epochs.')
flags.DEFINE_integer('n_classes', 1,
                     'Number of classes in case of classification.')
flags.DEFINE_integer('display_step', 10,
                     'Displaying step at training.')
flags.DEFINE_integer('n_layers', 3,
                     'Number of layers.')
flags.DEFINE_integer('n_neurons', 500,
                     'Number of Neurons.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def main(args):
    """The function for TF-Slim DNN model training.
    This function receives user-given parameters as gflag arguments. Then it
    creates the tensorflow model-graph, defines loss and optimizer. Finally,
    creates a training loop and saves the results and logs in the sub-directory.
    Args:
        args: This brings all gflags given user inputs with default values.
    Returns:
        None
    """

    data_x, data_y, c = data_providers.data_providers(FLAGS.x_data, FLAGS.y_data)

    X = data_x
    C = c
    T = data_y

    n = FLAGS.split_ratio
    fold = int(len(X) / 10)
    train_set = {}
    test_set = {}
    final_set = {}

    sa = SurvivalAnalysis()
    train_set['X'], train_set['T'], train_set['C'], train_set['A'] = sa.calc_at_risk(X[0:fold * 6, ], T[0:fold * 6], C[0:fold * 6]);

    n_obs = train_set['X'].shape[0]
    n_features = train_set['X'].shape[1]
    observed = 1 - train_set['C']

    # Start building the graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        logging.set_verbosity(tf.logging.INFO)

        if not tf.gfile.Exists(FLAGS.ckpt_dir):
            tf.gfile.MakeDirs(FLAGS.ckpt_dir)

        n_batches = int(n_obs / FLAGS.batch_size)
        decay_steps = int(FLAGS.n_epochs * n_batches)

        x = tf.placeholder("float", [None, n_features], name='features')
        a = tf.placeholder(tf.int32, [None], name='at_risk')
        o = tf.placeholder("float", [None], name='observed')

        # Create the model and pass the input values batch by batch
        hidden_layers = [FLAGS.n_neurons] * FLAGS.n_layers
        pred, end_points = dnn_model.multilayer_nn_model(x,  # x_batch
                                                         hidden_layers,
                                                         FLAGS.n_classes,
                                                         FLAGS.beta)
        # ----------------------------------------------------------------------
        tf.add_to_collection('pred', pred)
        # ----------------------------------------------------------------------

        global_step = get_or_create_global_step()

        lr = tf.train.exponential_decay(learning_rate=FLAGS.init_lr,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=FLAGS.lr_decay_rate,
                                        staircase=True, name='learning_rate')

        # Define loss
        cost = cox_layer.cost_function_observed(pred, a, o)
        tf.losses.add_loss(cost, loss_collection=tf.GraphKeys.LOSSES)
        total_loss = tf.losses.get_total_loss(name='add_loss')
        # ----------------------------------------------------------------------
        tf.add_to_collection('loss_collection', total_loss)
        # ----------------------------------------------------------------------
        tf.summary.scalar('loss', total_loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, name='optimizer').minimize(total_loss, global_step=global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            for epoch in range(FLAGS.n_epochs + 1):
                avg_cost = 0.0
                b_size = FLAGS.batch_size

                for i in range(n_batches - 1):
                    batch_x = train_set['X'][i * b_size:(i + 1) * b_size]
                    batch_a = train_set['A'][i * b_size:(i + 1) * b_size]
                    batch_o = observed[i * b_size:(i + 1) * b_size]

                    _, batch_cost, batch_pred = sess.run([optimizer,
                                                          total_loss,
                                                          pred],
                                                         feed_dict={x: batch_x, a: batch_a, o: batch_o})

                    avg_cost += batch_cost / n_batches

                if epoch % FLAGS.display_step == 0:
                    print("Epoch : ", "%05d" % (epoch + 1),
                          " cost = ", " {:.9f} ".format(avg_cost))

            saver.save(sess, FLAGS.ckpt_dir + 'deep_model', global_step=global_step)

    # Start building the graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        logging.set_verbosity(tf.logging.INFO)

        # Running a new session
        print("Starting 3rd session...")
        # This session is for validating the prediction.
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            new_saver = tf.train.import_meta_graph('./ckpt_dir/deep_model-81.meta')
            new_saver.restore(sess, './ckpt_dir/deep_model-81')

            x = graph.get_tensor_by_name("features:0")
            a = graph.get_tensor_by_name("at_risk:0")
            o = graph.get_tensor_by_name("observed:0")

            # Collection is a list of multiple variable names. So get 1st element.
            prediction = tf.get_collection('pred')[0]
            val_pred = sess.run(prediction, feed_dict={x: train_set['X']})
            print(val_pred)


if __name__ == '__main__':
    tf.app.run()
