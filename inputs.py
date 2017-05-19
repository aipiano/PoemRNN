import tensorflow as tf


def make_batch_from_record(file_name, batch_size, num_epochs):
    """
    see http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
    :param file_name: 
    :param batch_size: 
    :param num_epochs: 
    :return: batch of tokens, labels and length
    """
    filename_queue = tf.train.string_input_producer([file_name], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Define how to parse the example
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # Make batch. Note that dynamic_pad must be True.
    tokens, labels, lengths = tf.train.batch(
        [sequence_parsed['tokens'], sequence_parsed['labels'], context_parsed['length']],
        batch_size, num_threads=4, capacity=32 + batch_size*3, dynamic_pad=True
    )

    return tokens, labels, lengths




