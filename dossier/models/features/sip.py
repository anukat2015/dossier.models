'''extractor of noun phrases (NPs) and statistically interesting
phrases (SIPs) used to create features in FCs.

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.

'''
from __future__ import absolute_import, division, print_function

try:
    from collections import Counter, defaultdict
except ImportError:
    from backport_collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords

from dossier.models.features.stopwords import stopwords as dossier_stopwords


def sip_noun_phrases(tfidf, noun_phrases, limit=40):
    bow = tfidf[tfidf.id2word.doc2bow(noun_phrases)]
    return dict([(tfidf.id2word[word], count)
                 for word, count in Counter(dict(bow)).most_common(limit)])


def noun_phrases_as_tokens(text):
    '''Generate a bag of lists of unnormalized tokens representing noun
    phrases from ``text``.

    This is built around python's nltk library for getting Noun
    Phrases (NPs). This is all documented in the NLTK Book
    http://www.nltk.org/book/ch03.html and blog posts that cite the
    book.

    :rtype: list of lists of strings

    '''
    ## from NLTK Book:
    sentence_re = r'''(?x)      # set flag to allow verbose regexps
          ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
        | \w+(-\w+)*            # words with optional internal hyphens
        | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]      # these are separate tokens
    '''

    ## From Su Nam Kim paper:
    ## http://www.comp.nus.edu.sg/~kanmy/papers/10.1007_s10579-012-9210-3.pdf
    grammar = r'''
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    '''
    if len(text.strip()) == 0:
        return []

    chunker = nltk.RegexpParser(grammar)

    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)

    #print postoks
    tree = chunker.parse(postoks)
    stops = stopwords.words('english')
    stops += dossier_stopwords()

    ## These next four functions are standard uses of NLTK illustrated by
    ## http://alexbowe.com/au-naturale/
    ## https://gist.github.com/alexbowe/879414
    def leaves(tree):
        '''Finds NP (nounphrase) leaf nodes of a chunk tree.'''
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def acceptable_word(word):
        '''Checks conditions for acceptable word: length, stopword.'''
        return 2 <= len(word) <= 40 and word.lower() not in stops

    def get_terms(tree):
        for leaf in leaves(tree):
            yield [w for w,t in leaf if acceptable_word(w)]

    return list(get_terms(tree))


def noun_phrases(text, included_unnormalized=False):
    '''applies normalization to the terms found by noun_phrases_as_tokens
    and joins on '_'.

    :rtype: list of phrase strings with spaces replaced by ``_``.

    '''
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()

    def normalize(word):
        '''Normalises words to lowercase and stems and lemmatizes it.'''
        word = word.lower()
        try:
            word = stemmer.stem_word(word)
            word = lemmatizer.lemmatize(word)
        except:
            pass
        return word

    normalizations = defaultdict(list)
    for terms in noun_phrases_as_tokens(text):
        key = u'_'.join(map(normalize, terms))
        normalizations[key].append(u' '.join(terms))

    if included_unnormalized:
        return normalizations.keys(), normalizations
    else:
        return normalizations.keys()

