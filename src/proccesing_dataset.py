import re
import unicodedata
import tensorflow as tf
import io

# TODO add path to own dataset
PATH_DATASET = ''
PATH_DATASET_RESHAPE = ""



# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('<topic>')] for l in lines[:num_examples]]

    return zip(*word_pairs)


def get_data_set_mta(path_dataset, number_topics):
    max_example = 10000
    essay, topics = create_dataset(path_dataset, max_example)
    new_topics = []

    for topic in topics:
        token_topics = topic.split()
        # remove '<start> <end>'
        token_topics = token_topics[1:-1]
        unique = list(set(token_topics))

        if len(unique) > number_topics:
            new_topic_token = unique[0:number_topics]
            new_topic = " ".join(new_topic_token)
            new_topic = new_topic
            new_topics.append(new_topic)
        elif len(unique) < number_topics:

            new_topic_token = unique + (number_topics // len(unique) - 1) * unique
            new_topic_token = new_topic_token + unique[0:number_topics - len(new_topic_token)]

            new_topic = " ".join(new_topic_token)
            new_topic = new_topic
            new_topics.append(new_topic)

        else:
            new_topic = " ".join(unique)
            new_topic = new_topic
            new_topics.append(new_topic)

    essay_tensor, essay_lang_tokenizer = tokenize(essay)
    topics_tensor, topics_lang_tokenizer = tokenize(new_topics)

    return essay_tensor, essay_lang_tokenizer, topics_tensor, topics_lang_tokenizer


def reshape_dataset(max_len):
    lines_new = []
    with open(PATH_DATASET, 'r') as file_read:
        for line in file_read.readlines():
            essay, topics = line.split('<topic>')
            essay_processing = preprocess_sentence(essay)
            topics_new = topics.strip()

            essay_processing = essay_processing.split()[1:max_len + 1]
            new_essay = " ".join(essay_processing)

            new_line = new_essay + " <topic> " + topics_new + '\n'
            lines_new.append(new_line)

    with open(PATH_DATASET_RESHAPE, 'w') as write_file:
        for line in lines_new:
            write_file.write(line)


if __name__ == '__main__':
    essay_tensor, essay_lang_tokenizer, topics_tensor, topics_lang_tokenizer = get_data_set_mta(10)
    print(topics_tensor[0])
    reshape_dataset(300)
    # essay, topics = create_dataset(PATH_DATASET, 10000)

