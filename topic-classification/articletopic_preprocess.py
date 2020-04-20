"""
Code for preprocessing largely taken from this excellent student work: https://github.com/mmarinated/topic-modeling
Currently only implemented for English but can be adapted for other languages. See above repo for Hindi / Russian.
"""

import argparse
import csv
import json
import os
import re
import string
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords

csv.field_size_limit(sys.maxsize)

_digits_to_words_dict = {
    "english": {
        '0': ' zero',
        '1': ' one',
        '2': ' two',
        '3': ' three',
        '4': ' four',
        '5': ' five',
        '6': ' six',
        '7': ' seven',
        '8': ' eight',
        '9': ' nine',
    }
}

_common_forbidden_patterns = [
    #    "<!--.*?-->"  # this would remove any commented-out text, which is rare but sometimes substantial
    "\[\[category:.*?\]\]",  # EDITED remove categories [[Category:Far-left politics]]
    "\[\[категория:.*?\]\]",  # EDITED: remove category for Russian
    "\[\[श्रेणी:.*?\]\]",  # EDITED: remove category for Hindi
    "{{.*?}}"  # ? makes sure this is non-greedy and only matches to end of the template
    , "&amp;"
    , "&lt;"
    , "&gt;"
    , r"<ref[^<]*<\/ref>"
    , "<[^>]*>"
    , "\|left"
    , "\|\d+px"
    #     ,"\[\[category:"
    , r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b"
    , "\|thumb"
    , "\|right"
    , "\[\[image:[^\[\]]*"
    , "\[\[category:([^|\]]*)[^]]*\]\]"
    , "\[\[[a-z\-]*:[^\]]*\]\]"
    , "\["
    , "\]"
    , "\{[^\}]*\}"
    , r"\n"
    , " +"
]

_forbidden_patterns_dict = {
    "english": [
        r"[^a-zA-Z0-9 ]",
        r"\b[a-zA-Z]\b",
    ]
}


def _get_tokenizer_obj(language):
    if language == "english":
        from spacy.lang.en import English as Tokenizer
    else:
        raise NotImplementedError

    tokenizer = Tokenizer()
    return tokenizer


def _get_stopwords(language):
    try:
        # Download and set stop word list from NLTK
        nltk.download('stopwords')
        stop_words = set(stopwords.words(language))
    except OSError:
        raise NotImplementedError
    return stop_words


class Parser:
    def __init__(self, language):
        """
        Parameters:
        -----------
        language : str, "english" or "russian" or "hindi"
        """
        self.LANGUAGE = language
        ## SAME for all languages
        # Load punctuations
        self.PUNCTUATIONS = set(string.punctuation)

        ## SPECIFIC to language
        # Patterns for regex
        self.PATTERNS = _common_forbidden_patterns + _forbidden_patterns_dict[language]
        self.PATTERNS = [re.compile(p) for p in self.PATTERNS]
        # Digits to words
        self.DIGITS_TO_WORDS_DICT = _digits_to_words_dict[language]
        # Load tokenizer
        self.TOKENIZER = _get_tokenizer_obj(language)
        # Set stop word list
        self.STOP_WORDS = _get_stopwords(language)

    def _clean_patterns(self, text):
        """ Clean text using regex - similar to what is used in FastText paper. """
        for p in self.PATTERNS:
            text = re.sub(p, ' ', text)
        return text

    def _substitute_digits_with_words(self, text):
        """ Convert digits to their names. """
        chars = text.strip()
        new_sentence = [self.DIGITS_TO_WORDS_DICT.get(char, char) for char in chars]
        return ''.join(new_sentence)

    def _tokenize(self, sent):
        """ Lowercase and remove punctuation. """
        tokens = self.TOKENIZER(sent)
        return [token.text for token in tokens if (token.text not in self.PUNCTUATIONS)]

    def _remove_empty_tokens(self, tokens):
        """ Removes empty tokens that consist of several spaces. """
        return [token for token in tokens if not token.strip() == '']

    def _remove_stop_words(self, tokens):
        """ Removes stop words. """
        return [token for token in tokens if not token in self.STOP_WORDS]

    def preprocess_pipeline(self, wikitext):
        """ Combines all text transformations in a pipeline and returns list of tokens. """
        wikitext = str(wikitext).lower()
        wikitext = self._clean_patterns(wikitext)
        wikitext = self._substitute_digits_with_words(wikitext)
        wikitext = self._tokenize(wikitext)
        wikitext = self._remove_empty_tokens(wikitext)
        wikitext = self._remove_stop_words(wikitext)
        return ' '.join(wikitext)


def to_dataframe(data_fn):
    """Parses Wikidata claims for fastText processing"""
    print("Converting {0} -> fastText format.".format(data_fn))
    df = pd.read_json(data_fn, lines=True)
    parser = Parser(language="english")
    df['text'] = df['text'].apply(parser.preprocess_pipeline)
    df['taxo_labels'] = df['taxo_labels'].apply(lambda x: list(set(x)))
    data = df.sample(frac=1, replace=False)
    print(data.info())
    print(data.head(n=5))
    return data

def wikidata_to_fasttext(data, fasttext_datafn, fasttext_readme):
    """Write xy-data and associated metadata to respective files."""
    metadata = {}
    potential_metadata_cols = [c for c in data.columns if c not in ('text', 'taxo_labels')]
    if potential_metadata_cols:
        print("Metadata columns: {0}".format(potential_metadata_cols))
        for idx, row in data.iterrows():
            metadata[idx] = {}
            for c in potential_metadata_cols:
                metadata[idx][c] = row[c]
    write_fasttext(data['text'], fasttext_datafn, fasttext_readme, data["taxo_labels"], metadata)


def write_fasttext(x_data, data_fn, readme_fn, y_data=None, metadata={}):
    """Write data in fastText format."""
    written = 0
    skipped = 0
    no_text = 0
    with open(readme_fn, 'w') as readme_fout:
        with open(data_fn, 'w') as data_fout:
            for idx, wikitext in x_data.iteritems():
                if not len(wikitext):
                    no_text += 1
                    continue
                lbls = y_data.loc[idx]
                if len(lbls):
                    mlcs = ' '.join(['__label__{0}'.format(c.replace(" ", "_")) for c in lbls])
                    data_fout.write("{0} {1}\n".format(mlcs, wikitext))
                else:
                    skipped += 1
                    continue
                if metadata:
                    readme_fout.write(json.dumps(metadata[idx]) + '\n')
                written += 1
    print("{0} data points written to {1} and {2}. {3} skipped and {4} w/o claims.".format(written, data_fn, readme_fn,
                                                                                               skipped, no_text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fn")
    parser.add_argument("--output_folder")
    parser.add_argument("--train_prop", type=float, default=0.8)
    parser.add_argument("--val_prop", type=float, default=0.1)
    parser.add_argument("--test_prop", type=float, default=0.1)
    args = parser.parse_args()

    assert args.train_prop + args.val_prop + args.test_prop == 1

    data = to_dataframe(args.data_fn)
    base_fn = os.path.join(args.output_folder, os.path.splitext(os.path.basename(args.data_fn))[0])

    if args.train_prop == 1:
        fasttext_datafn = '{0}_{1}_data.txt'.format(base_fn, len(data))
        fasttext_readme = '{0}_{1}_meta.txt'.format(base_fn, len(data))
        wikidata_to_fasttext(data, fasttext_datafn, fasttext_readme)

    else:
        train_idx = int(len(data) * args.train_prop)
        val_idx = train_idx + int(len(data) * args.val_prop)

        if train_idx > 0:
            train_data = data[:train_idx]
            print("{0} training datapoints.".format(len(train_data)))
            fasttext_datafn = '{0}_{1}_train_data.txt'.format(base_fn, len(train_data))
            fasttext_readme = '{0}_{1}_train_meta.txt'.format(base_fn, len(train_data))
            wikidata_to_fasttext(train_data, fasttext_datafn, fasttext_readme)

        if val_idx > train_idx:
            val_data = data[train_idx:val_idx]
            print("{0} validation datapoints.".format(len(val_data)))
            fasttext_datafn = '{0}_{1}_val_data.txt'.format(base_fn, len(val_data))
            fasttext_readme = '{0}_{1}_val_meta.txt'.format(base_fn, len(val_data))
            wikidata_to_fasttext(val_data, fasttext_datafn, fasttext_readme)

        if val_idx < len(data):
            test_data = data[val_idx:]
            print("{0} test datapoints.".format(len(test_data)))
            fasttext_datafn = '{0}_{1}_test_data.txt'.format(base_fn, len(test_data))
            fasttext_readme = '{0}_{1}_test_meta.txt'.format(base_fn, len(test_data))
            wikidata_to_fasttext(test_data, fasttext_datafn, fasttext_readme)


if __name__ == "__main__":
    main()