import tensorflow as tf
import numpy as np

poems_file = 'poems.txt'


def main():
    with open(poems_file, mode='r', encoding='utf-8') as f:
        poems = f.readlines()
    for i, poem in enumerate(poems):
        poem = poem.replace(' ', '')
        poem = poem.replace('\n', '')
        poems[i] = poem

    char_ids = build_character_ids(poems)
    poem_vectors = poems2vectors(poems, char_ids)
    np.random.shuffle(poem_vectors)

    writer = tf.python_io.TFRecordWriter('./examples.record')
    for vector in poem_vectors:
        labels = vector[:]
        labels[: -1] = vector[1:]
        example = make_example(vector, labels)
        writer.write(example.SerializeToString())
    writer.close()


def make_example(sequence, labels):
    """
    see http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
    :param sequence: 
    :param labels: 
    :return: 
    """
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


def build_character_ids(poems):
    char_dict = {}
    for poem in poems:
        for char in poem:
            if char_dict.get(char) is None:
                char_dict[char] = 1
                continue
            char_dict[char] += 1

    sorted_items = sorted(char_dict.items(), key=lambda i: i[1], reverse=True)
    # The character id is start from 1, cause 0 is reserved for padding.
    char_ids = {item[0]: i+1 for i, item in enumerate(sorted_items)}

    # Save statistic result
    report = ['Number of characters: %d\n' % len(char_dict)]
    for char, statistic in sorted_items:
        report.append('%s: %d\n' % (char, statistic))
    f = open('./statistic.txt', mode='w', encoding='utf-8')
    f.writelines(report)
    f.close()

    return char_ids


def poems2vectors(poems, char_ids):
    poem_vectors = []
    for poem in poems:
        poem_vector = list(map(lambda c: char_ids[c], poem))
        poem_vectors.append(poem_vector)
    return poem_vectors


if __name__ == '__main__':
    main()
