### Topic Classification
There are three scripts in this directory. A quick overview of each is described below:
* `articletopic_preprocess.py`: this file takes article text + labels and converts it into train/val/test files suitable
for fastText and model evaluation. Missing right now is the script that creates these input files, but they are just 
one JSON object per line where each JSON object represents an article and has the article's wikitext from the dumps and
the articles' topic labels (see: https://figshare.com/articles/Wikipedia_Articles_and_Associated_WikiProject_Templates/10248344)
* `articletopic_model.py`: this takes the output of `articletopic_preprocess.py` and builds either fastText models for
topic classification or various types of fasttext-like models in Keras.
* `articletopic_fasttext_comprehensive.py`: this has similar code to `articletopic_model.py` but is focused solely on the 
fastText model and has more comprehensive grid-search / model analysis.