# some util functions
import nltk


def write_file(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as fout:
        for item in data:
            fout.write('{}\n'.format(item))


def split_sentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences