import tensorflow as tf
import numpy as np
import model


flags = tf.app.flags
flags.DEFINE_string('statistic_file', './data/statistic.txt', 'The path of statistic file.')
flags.DEFINE_integer('num_units', 128, 'Number of units in an rnn cell')
flags.DEFINE_integer('num_layers', 3, 'Number of layers of the whole rnn network')
flags.DEFINE_integer('num_examples', 10, 'Number of examples to generate')
FLAGS = flags.FLAGS


def main(_):
    char2id, id2char = load_char_dict(FLAGS.statistic_file)
    num_chars = len(char2id)
    start_id = char2id['[']
    stop_id = char2id[']']
    tokens = np.array(list(range(0, num_chars+1)), dtype=np.int64)    # 0 is used for padding

    with tf.Graph().as_default():
        inputs_holder = tf.placeholder(tf.int64, [1, 1])
        rnn = model.DyCharRNN(inputs_holder, num_chars, None, num_units=FLAGS.num_units,
                              num_layers=FLAGS.num_layers, batch_size=1)
        probs = rnn.get_probabilities()

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            latest_ckpt = tf.train.latest_checkpoint('./checkpoints')
            if latest_ckpt is not None:
                print('restore checkpoint : %s.' % latest_ckpt)
                saver.restore(sess, latest_ckpt)
            else:
                print('No model found.')
                return

            for i in range(FLAGS.num_examples):
                state = sess.run(rnn.cell.zero_state(1, dtype=tf.float32))
                last_token = np.array([start_id], dtype=np.int64)
                poem = '['

                while poem[-1] != ']':
                    p, state = sess.run([probs, rnn.last_state],
                                        feed_dict={inputs_holder: [last_token], rnn.initial_state: state})
                    last_token = np.random.choice(tokens, 1, p=p[0])
                    poem += id2char[last_token[0]]
                print(poem)


def load_char_dict(statistic_file):
    statistic = open(statistic_file, mode='r', encoding='utf-8')
    lines = statistic.readlines()[1:]   # exclude the first line.
    char2id = {}
    id2char = {}
    for i, line in enumerate(lines):
        char = line.split(':')[0]
        char2id[char] = i+1
        id2char[i+1] = char
    return char2id, id2char


if __name__ == '__main__':
    tf.app.run(main)
