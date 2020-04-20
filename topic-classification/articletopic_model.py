import argparse
import csv
import multiprocessing
import os
import sys

csv.field_size_limit(sys.maxsize)

import fasttext
from keras.initializers import Constant
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

BATCH_SIZE = 32
MAX_LEN = 500
MAXTHREADS = min(16, multiprocessing.cpu_count() - 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='fastText',
                        help="fastText or Keras")
    parser.add_argument("--output_model",
                        default=None,
                        help="Path to save model. Leave empty if you don't want to save model.")
    # Model Parameters -- fastText and Keras
    parser.add_argument("--training_data",
                        default="/home/isaacj/topic_models/enwiki.balanced_article_sample.w_article_text_50413_train_data.txt",
                        help="Path to training data in fastText format.")
    parser.add_argument("--val_data",
                        default="/home/isaacj/topic_models/enwiki.balanced_article_sample.w_article_text_6301_val_data.txt",
                        help="Path to validation data in fastText format.")
    parser.add_argument("--test_data",
                        default="/home/isaacj/topic_models/enwiki.balanced_article_sample.w_article_text_6303_test_data.txt",
                        help="Path to test data in fastText format.")
    parser.add_argument("--word_vectors",
                        default='/home/isaacj/topic_models/enwiki.vectors.20191201.skipgram_50.300k.vec',
                        help="Path to pre-trained word vectors to use in model.")
    parser.add_argument("--lr",
                        default=0.1, type=float,
                        help="Learning rate.")
    parser.add_argument("--ndims",
                        default=50, type=int,
                        help="Dimensionality of word embeddings and any hidden layers.")
    parser.add_argument("--wordngrams",
                        default=1, type=int,
                        help="1 = unigram; 2 = bigram; etc.")
    parser.add_argument("--epochs",
                        default=25, type=int,
                        help="# of passes through training data.")
    # Model Parameters -- fastText only
    parser.add_argument("--minCount",
                        default=1000000, type=int,
                        help="fastText only: minimum number of occurrences for word to be included. "
                             "If pretrained word embeddings are used, all of those words will be retained. "
                             "Set very high to only use pre-trained word vectors in fastText")
    parser.add_argument("--ws",
                        default=20, type=int,
                        help="fastText only: window-size -- unclear to me how this is used in supervised model")
    # Model Parameters -- Keras only
    parser.add_argument("--max_features",
                        default=300000, type=int,
                        help="Keras only: maximum number of words to include in model.")
    parser.add_argument("--finetune_embs",
                        default=False, action="store_true",
                        help="Keras only: if yes, fine-tune word embeddings. Otherwise, freeze them.")
    parser.add_argument("--num_hidden_layer",
                        default=0, type=int,
                        help="Keras only: # of additional hidden layers between average embeddings and logistic regression.")
    parser.add_argument("--use_ft_pretrained",
                        default=False, action="store_true",
                        help="Keras only: use fine-tuned embeddings from a fastText model.")

    args = parser.parse_args()

    if not os.path.exists(args.word_vectors):
        print("No pretrained word vectors.")
        args.word_vectors = ""

    if args.model_type == 'fastText':
        model = fastTextModel(args)
    elif args.model_type == 'keras':
        model = KerasModel(args)
    else:
        raise NotImplementedError("Did not recognize model type:", args.model_type)

    model.build_model()
    model.test_model()

    if args.output_model:
        model.save_model(args.output_model)

class KerasModel():

    def __init__(self, args):
        self.use_ft_pretrained = args.use_ft_pretrained
        self.word_vectors = args.word_vectors
        self.max_features = args.max_features
        self.training_data = args.training_data
        self.val_data = args.val_data
        self.test_data = args.test_data
        self.wordngrams = args.wordngrams
        self.ndims = args.ndims
        self.finetune_embs = args.finetune_embs
        self.num_hidden_layer = args.num_hidden_layer
        self.epochs = args.epochs

        # hacky but I do this when initializing so I don't have to store all the fastText args
        print("Loading word vectors")
        if self.use_ft_pretrained:
            ft_model = fastTextModel(args)
            ft_model.build_model()
            self.embeddings, self.vocab_to_idx = ft_model.extract_embeddings()
        else:
            self.embeddings, self.vocab_to_idx = load_wvs(self.word_vectors, self.max_features)
        print("{0} words of {1} dimensions.".format(len(self.vocab_to_idx), self.embeddings.shape[1]))

        print("Building label dictionary.")
        self.lbl_to_idx = self.get_lbl_indices(self.training_data)
        print("{0} labels.".format(len(self.lbl_to_idx)))


    def build_model(self):
        """ Based on https://keras.io/examples/imdb_fasttext/ """
        print('Loading data...')
        x_train, y_train = self.prepare_data(self.training_data, label='train')
        x_val, y_val = self.prepare_data(self.val_data, label='val')

        print('Build model...')
        self.model = Sequential()

        # we start off with an embedding layer which maps tokens -> k-dimensional embeddings
        if os.path.exists(self.word_vectors):
            print("Using pretrained embeddings:", self.word_vectors)
            embedding_layer = Embedding(len(self.embeddings),
                                        self.ndims,
                                        input_length=MAX_LEN,
                                        embeddings_initializer=Constant(self.embeddings),
                                        trainable=self.finetune_embs)

        else:
            print("Starting with random embeddings.")
            embedding_layer = Embedding(self.max_features,
                                        self.ndims,
                                        input_length=MAX_LEN,
                                        trainable=self.finetune_embs)
        self.model.add(embedding_layer)

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        self.model.add(GlobalAveragePooling1D())

        # optionally add a hidden layer (same dimensionality as embeddings)
        # this is mainly for if the embeddings are set to not trainable
        for _ in range(self.num_hidden_layer):
            self.model.add(Dense(self.ndims))
            self.model.add(Dropout(0.25))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        self.model.add(Dense(len(self.lbl_to_idx), activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        print("Model:\n", self.model.summary())

        # train model
        self.model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=self.epochs,
                       validation_data=(x_val, y_val))

    def prepare_data(self, data_fn, label=""):
        """Convert fastText-formatted data / labels file to Keras format."""
        x_data = []
        y_data = []
        with open(data_fn, 'r') as fin:
            for line in fin:
                labels, tokens = parse_ft_line(line, self.vocab_to_idx, self.lbl_to_idx)
                labels_onehot = np.zeros(len(self.lbl_to_idx))
                for l_idx in labels:
                    labels_onehot[l_idx] = 1
                y_data.append(labels_onehot)
                x_data.append(tokens)

        print(len(x_data), '{0} sequences'.format(label))
        print('Average {0} sequence length: {1}'.format(label, np.mean(list(map(len, x_data)), dtype=int)))

        if self.wordngrams > 1:
            print('Adding {}-gram features'.format(self.wordngrams))
            # Create set of unique n-gram from the training set.
            if label == 'train':
                ngram_set = set()
                for input_list in x_data:
                    for i in range(2, self.wordngrams + 1):
                        set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                        ngram_set.update(set_of_ngram)

                # Dictionary mapping n-gram token to a unique integer.
                # Integer values are greater than max_features in order
                # to avoid collision with existing features.
                start_index = self.max_features + 1
                self.token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
                indice_token = {self.token_indice[k]: k for k in self.token_indice}

                # max_features is the highest integer that could be found in the dataset.
                self.max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting x_train and x_test with n-grams features
            x_data = self.add_ngram(x_data, self.token_indice, self.wordngrams)
            print('Average {0} sequence length: {1}'.format(label, np.mean(list(map(len, x_data)), dtype=int)))

        # Keras expects fixed-length input so articles are trimmed to first MAX_LEN tokens
        print('Pad sequences (samples x time)')
        x_data = sequence.pad_sequences(x_data, maxlen=MAX_LEN, padding='post', truncating='post')
        print('x_{0} shape: {1}'.format(label, x_data.shape))

        y_data = np.asarray(y_data)
        print('y_{0} shape: {1}'.format(label, y_data.shape))

        return x_data, y_data


    def extract_embeddings(self):
        return self.embeddings, self.vocab_to_idx

    def get_lbl_indices(self, data_fn):
        """Generate mapping of label -> integer for Keras."""
        lbl_to_idx = {}
        with open(data_fn, 'r') as fin:
            for line in fin:
                line = line.strip().split()
                for w in line:
                    if w.startswith('__label__') and w not in lbl_to_idx:
                        lbl_to_idx[w] = len(lbl_to_idx)
        return lbl_to_idx

    def create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(self, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.

        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def test_model(self, threshold=0.5):
        """Keras model statistics."""
        # build statistics dataframe for printing
        x_test, y_test = self.prepare_data(self.test_data, label='test')

        print("==== test statistics ====")
        lbl_statistics = {}
        idx_to_lbl = {i: l for l, i in self.lbl_to_idx.items()}
        for lbl in self.lbl_to_idx:
            lbl_statistics[lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'true': [], 'pred': []}
        predictions = self.model.predict(x_test, batch_size=BATCH_SIZE)
        for i in range(len(predictions)):
            predicted_labels = []
            actual_labels = [idx_to_lbl[idx] for idx, lbl in enumerate(y_test[i]) if lbl == 1]
            for idx in range(len(predictions[i])):
                prob = predictions[i][idx]
                lbl = idx_to_lbl[idx]
                lbl_statistics[lbl]['true'].append(y_test[i][idx])
                lbl_statistics[lbl]['pred'].append(prob)
                if prob > threshold:
                    predicted_labels.append(lbl)
            for lbl in self.lbl_to_idx:
                if lbl in actual_labels and lbl in predicted_labels:
                    lbl_statistics[lbl]['n'] += 1
                    lbl_statistics[lbl]['TP'] += 1
                elif lbl in actual_labels:
                    lbl_statistics[lbl]['n'] += 1
                    lbl_statistics[lbl]['FN'] += 1
                elif lbl in predicted_labels:
                    lbl_statistics[lbl]['FP'] += 1
                else:
                    lbl_statistics[lbl]['TN'] += 1
        print_lbl_stats(lbl_statistics)

    def save_model(self, fp):
        self.model.save(fp)

class fastTextModel():

    def __init__(self, args):
        self.word_vectors = args.word_vectors
        self.max_features = args.max_features
        self.training_data = args.training_data
        self.val_data = args.val_data
        self.test_data = args.test_data
        self.wordngrams = args.wordngrams
        self.lr = args.lr
        self.ws = args.ws
        self.minCount = args.minCount
        self.ndims = args.ndims
        self.epochs = args.epochs

    def build_model(self):
        """Build standard fastText supervised model."""
        print("Building fasttext model: {0} lr; {1} min count; {2} epochs; {3} ws; wv: {4}".format(
            self.lr, self.minCount, self.epochs, self.ws, self.word_vectors))
        self.model = fasttext.train_supervised(input=self.training_data,
                                               minCount=self.minCount,
                                               wordNgrams=self.wordngrams,
                                               lr=self.lr,
                                               epoch=self.epochs,
                                               pretrainedVectors=self.word_vectors,  # these cannot be frozen
                                               ws=self.ws,
                                               dim=self.ndims,
                                               minn=0,  # no subwords
                                               maxn=0,  # no subwords
                                               thread=MAXTHREADS,  # number of CPUs to use in training
                                               loss='ova',  # one vs. all regression
                                               verbose=2)

    def extract_embeddings(self):
        """Extract embeddings from a trained fastText model."""
        print("{0} words in fastText model.".format(len(self.model.words)))
        vocab_to_idx = {}
        embeddings = np.zeros((len(self.model.words) + 1, self.model.get_dimension()))
        for idx, word in enumerate(self.model.words):
            dims = self.model.get_word_vector(word)
            embeddings[idx] = dims
            vocab_to_idx[word] = idx
        return embeddings, vocab_to_idx

    def test_model(self):
        # build statistics dataframe for printing
        print("==== test statistics ====")
        lbl_statistics = {}
        threshold = 0.5
        for lbl in self.model.labels:
            lbl_statistics[lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'true': [], 'pred': []}
        with open(self.test_data, 'r') as fin:
            for line in fin:
                topics, _ = parse_ft_line(line)
                prediction = self.model.predict(line.strip(), k=-1)
                predicted_labels = []
                for idx in range(len(prediction[0])):
                    prob = prediction[1][idx]
                    lbl = prediction[0][idx]
                    lbl_statistics[lbl]['true'].append(int(lbl in topics))
                    lbl_statistics[lbl]['pred'].append(prob)
                    if prob > threshold:
                        predicted_labels.append(lbl)
                for lbl in self.model.labels:
                    if lbl in topics and lbl in predicted_labels:
                        lbl_statistics[lbl]['n'] += 1
                        lbl_statistics[lbl]['TP'] += 1
                    elif lbl in topics:
                        lbl_statistics[lbl]['n'] += 1
                        lbl_statistics[lbl]['FN'] += 1
                    elif lbl in predicted_labels:
                        lbl_statistics[lbl]['FP'] += 1
                    else:
                        lbl_statistics[lbl]['TN'] += 1

        print_lbl_stats(lbl_statistics)


    def save_model(self, fp):
        print("Dumping fasttext model to {0}".format(fp))
        self.model.save_model(fp)

def parse_ft_line(line, vocab_to_idx=None, lbl_to_idx=None):
    """Convert line from fastText format to list of labels and tokens.

    If vocab_to_idx and/or lbl_to_idx provided, tokens are converted to integer indices.
    """
    line = line.strip().split()
    labels = [w for w in line if w.startswith('__label__')]
    if lbl_to_idx:
        labels = [lbl_to_idx[w] for w in labels]
    tokens = [w for w in line if not w.startswith('__label__')]
    if vocab_to_idx:
        tokens = [vocab_to_idx.get(w, len(vocab_to_idx)) for w in tokens]
    return labels, tokens

def load_wvs(wv_fn, max_features=None):
    """Load in word vectors.

    The final embedding is an out-of-vocab embedding of just zeros in line with fastText.

    Expected format is space-separated plaintext file:
        Line 1: <number of words (n)> <dimensionality of embeddings (k)>
        Line 2 to n+1: <word> <dim1> <dim2> ... <dim k>
    """
    vocab_to_idx = {}
    with open(wv_fn, 'r') as fin:
        vocab_count, dimensions = [int(v) for v in next(fin).strip().split()]
        if max_features:
            vocab_count = min(vocab_count, max_features)
        embeddings = np.zeros((vocab_count + 1, dimensions))
        for idx, line in enumerate(fin):
            if idx == max_features:
                break
            line = line.strip().split()
            w = line[0]
            dims = np.asarray([float(v) for v in line[1:]])
            embeddings[idx] = dims
            vocab_to_idx[w] = idx
    return embeddings, vocab_to_idx

def print_lbl_stats(lbl_statistics):
    """Print model performance statistics."""
    for lbl in lbl_statistics:
        s = lbl_statistics[lbl]
        fpr, tpr, _ = roc_curve(s['true'], s['pred'])
        s['pr-auc'] = auc(fpr, tpr)
        s['avg_pre'] = average_precision_score(s['true'], s['pred'])
        try:
            s['precision'] = s['TP'] / (s['TP'] + s['FP'])
        except ZeroDivisionError:
            s['precision'] = 0
        try:
            s['recall'] = s['TP'] / (s['TP'] + s['FN'])
        except ZeroDivisionError:
            s['recall'] = 0
        try:
            s['f1'] = 2 * (s['precision'] * s['recall']) / (s['precision'] + s['recall'])
        except ZeroDivisionError:
            s['f1'] = 0
    print("\n=== Mid Level Categories ===")
    mlc_statistics = pd.DataFrame(lbl_statistics).T
    mlc_statistics['mid-level-category'] = [s.replace('__label__', '').replace('_', ' ') for s in mlc_statistics.index]
    mlc_statistics.set_index('mid-level-category', inplace=True)
    mlc_statistics[''] = '-->'
    mlc_statistics = mlc_statistics[['n', '', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1', 'pr-auc', 'avg_pre']]
    with pd.option_context('display.max_rows', None):
        print(mlc_statistics)
    print("\nPrecision: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['precision'], weights=mlc_statistics['n']), np.mean(mlc_statistics['precision'])))
    print("Recall: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['recall'], weights=mlc_statistics['n']), np.mean(mlc_statistics['recall'])))
    print("F1: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['f1'], weights=mlc_statistics['n']), np.mean(mlc_statistics['f1'])))
    print("PR-AUC: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['pr-auc'], weights=mlc_statistics['n']), np.mean(mlc_statistics['pr-auc'])))
    print("Avg pre.: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['avg_pre'], weights=mlc_statistics['n']), np.mean(mlc_statistics['avg_pre'])))


if __name__ == "__main__":
    main()
