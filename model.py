import tensorflow as tf
rnn_cell = tf.nn.rnn_cell
seq2seq = tf.nn.seq2seq


class DyCharRNN:
    def __init__(self, inputs, num_chars, sequence_lengths=None, cell_type='lstm', num_units=128, num_layers=2, batch_size=64, embedding_size=None):
        """
        A dynamic character-level RNN
        :param inputs: character vectors(without embedding) with shape [batch_size, max_time_steps]
        :param sequence_lengths: A 1-D tensor with shape [batch_size] specifies length of each sequence in the inputs
        :param num_chars: total characters the RNN can learn
        :param cell_type: lstm, gru or rnn
        :param num_units: number of units in a cell
        :param num_layers: 
        :param batch_size: 
        :param embedding_size: vector size after embedding
        """
        self.batch_size = batch_size
        num_chars += 1  # reserved a character (id == 0) for padding

        if cell_type == 'lstm':
            cell_func = rnn_cell.BasicLSTMCell
        elif cell_type == 'gru':
            cell_func = rnn_cell.GRUCell
        elif cell_type == 'rnn':
            cell_func = rnn_cell.BasicRNNCell
        else:
            raise Exception('Unsupported cell type: {}'.format(cell_type))

        cells = []
        for i in range(num_layers):
            cells.append(cell_func(num_units))  # output_size = num_units

        self.cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('RNN'):
            initializer = tf.truncated_normal_initializer()
            softmax_w = tf.get_variable('softmax_w', [num_units, num_chars], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [num_chars], initializer=initializer)
            with tf.device("/cpu:0"):
                # embedding_size is the true size input to the first rnn cell.
                embedding_size = embedding_size or num_units

                # each row is an embedding of a character
                embedding = tf.get_variable('embedding', [num_chars, embedding_size], initializer=initializer)

                # dense_inputs has a shape of [batch_size, max_time_steps, embedding_size]
                dense_inputs = tf.nn.embedding_lookup(embedding, inputs)

        # each batch has different length (max_time_steps), so we use dynamic_rnn.
        outputs, self.last_state = tf.nn.dynamic_rnn(self.cell, dense_inputs, sequence_length=sequence_lengths,
                                                     initial_state=self.initial_state, scope='DyCharRNN')
        # outputs above has a shape of [batch_size, max_time_steps, num_units], /
        # we change it to [-1, num_units] for convenience.
        outputs = tf.reshape(outputs, [-1, num_units])
        self.logits = tf.matmul(outputs, softmax_w) + softmax_b

    def get_loss(self, targets):
        """
        :param targets: A Tensor with the same shape as RNN's inputs (e.g. [batch_size, max_time_steps])
        :return: loss
        """
        targets = tf.reshape(targets, [-1])
        # for this sparse-version function, the targets are the indices of labels rather than one-hot-encoded vectors.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, targets)

        mask = tf.sign(tf.to_float(targets))  # in targets, 0 represent for a padding which should be mask out.
        losses *= mask  # mask out the loss of paddings
        total_loss = tf.reduce_sum(losses) / tf.reduce_sum(mask)    # divide by the true length of characters
        return total_loss

    def get_probabilities(self):
        return tf.nn.softmax(self.logits)








