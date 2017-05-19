import tensorflow as tf
import inputs
import model
import time
from os.path import join

flags = tf.app.flags
flags.DEFINE_string('statistic_file', './data/statistic.txt', 'The path of statistic file.')
flags.DEFINE_string('example_record', './data/examples.record', 'The path of a record file which stores training examples.')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('grad_clip', 5.0, 'Gradient clipping value')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to training.')
flags.DEFINE_boolean('reload_model', True, 'Load model from the latest checkpoint file if it exist.')
flags.DEFINE_integer('num_units', 128, 'Number of units in an rnn cell')
flags.DEFINE_integer('num_layers', 3, 'Number of layers of the whole rnn network')
FLAGS = flags.FLAGS


def train(_):
    num_chars = get_num_characters(FLAGS.statistic_file)
    with tf.Graph().as_default():
        # build the network and loss
        tokens, labels, lengths = inputs.make_batch_from_record(FLAGS.example_record, FLAGS.batch_size,
                                                                FLAGS.num_epochs)
        rnn = model.DyCharRNN(tokens, num_chars, lengths, num_units=FLAGS.num_units, num_layers=FLAGS.num_layers,
                              batch_size=FLAGS.batch_size)
        loss = rnn.get_loss(labels)

        # build training operator
        global_step = tf.Variable(0, trainable=False, name='global_step')
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, name='Adam')
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        # build summary
        tf.summary.scalar('loss', loss)
        merge_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            if FLAGS.reload_model:
                latest_ckpt = tf.train.latest_checkpoint('./checkpoints')
                if latest_ckpt is not None:
                    print('restore checkpoint : %s.' % latest_ckpt)
                    saver.restore(sess, latest_ckpt)

            step = 0
            try:
                while not coord.should_stop():
                    sec_per_batch = time.time()
                    sess.run(rnn.initial_state)  # zero state for every iterations.
                    _, batch_loss, step = sess.run([train_op, loss, global_step])

                    if step % 10 == 0:
                        batches_per_sec = 1.0 / (time.time() - sec_per_batch)
                        print('#iteration %d. loss_c: %.3f. (%.3f batches/sec)' % (step, batch_loss, batches_per_sec))

                    if step % 30 == 0:
                        summary = sess.run(merge_summary)
                        summary_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        saver.save(sess, join('./checkpoints', 'model.ckpt'), step)

            except tf.errors.OutOfRangeError:
                print('reach num_epochs. training finished.')
                print('saving final checkpoints...')
                saver.save(sess, join('./checkpoints', 'model.ckpt'), step)
            finally:
                summary = sess.run(merge_summary)
                summary_writer.add_summary(summary, step)
                coord.request_stop()  # stop all threads
                print('done!')

            coord.join(threads)  # wait threads to exit


def get_num_characters(statistic_file):
    statistic = open(statistic_file, mode='r', encoding='utf-8')
    first_line = statistic.readline()
    _, num_chars = first_line.split(':')
    return int(num_chars)


if __name__ == '__main__':
    tf.app.run(train)
