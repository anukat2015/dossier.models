from __future__ import absolute_import, division, print_function
import os

try:
    from gensim import corpora, models
    TFIDF = True
except ImportError:
    TFIDF = False
import pytest

from dossier.models.features import sip
from dossier.models.tests import kvl, store


@pytest.fixture
def tfidf():
    if not TFIDF:
        return
    doc1 = u'Andrew likes Diet Pepsi.'
    doc2 = u'Andrew knows the muffin man.'
    doc3 = u'Andrew lives near the muffin man on Shirley Lane.'
    corpus = map(sip.noun_phrases, [doc1, doc2, doc3])
    dictionary = corpora.Dictionary(corpus)
    bows = [dictionary.doc2bow(tokens) for tokens in corpus]
    return models.TfidfModel(bows, id2word=dictionary)

@pytest.fixture
def tfidf_path():
    dir_name = os.path.dirname(__file__)
    return os.path.join(dir_name, 'tfidf_model.dump')

# def test_fc_generator(tfidf):
    # print(tfidf)
    # print(tfidf[u'Andrew likes Diet Pepsi'])
    # assert False
