from __future__ import absolute_import, division, print_function

from collections import Counter

import nltk
from nltk.corpus import stopwords

from dossier.models.features.stopwords import stopwords as dossier_stopwords


def sip_noun_phrases(tfidf, noun_phrases, limit=40):
    bow = tfidf[tfidf.id2word.doc2bow(noun_phrases)]
    return {tfidf.id2word[word]: count
            for word, count in Counter(dict(bow)).most_common(limit)}


def noun_phrases(text):
    '''Generate a bag of normalized stemmed noun phrases from ``text``.

    This is built around python's nltk library for getting Noun
    Phrases (NPs). This is all documented in the NLTK Book
    http://www.nltk.org/book/ch03.html and blog posts that cite the
    book.

    :rtype: list of phrase strings with spaces replaced by ``_``.
    '''
    ## from NLTK Book:
    sentence_re = r'''(?x)      # set flag to allow verbose regexps
          ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
        | \w+(-\w+)*            # words with optional internal hyphens
        | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]      # these are separate tokens
    '''

    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()

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

    def normalise(word):
        '''Normalises words to lowercase and stems and lemmatizes it.'''
        word = word.lower()
        word = stemmer.stem_word(word)
        word = lemmatizer.lemmatize(word)
        return word

    def acceptable_word(word):
        '''Checks conditions for acceptable word: length, stopword.'''
        return 2 <= len(word) <= 40 and word.lower() not in stops

    def get_terms(tree):
        for leaf in leaves(tree):
            yield [normalise(w) for w,t in leaf if acceptable_word(w)]

    return ['_'.join(term) for term in get_terms(tree)]
