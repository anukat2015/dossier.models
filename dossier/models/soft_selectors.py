'''Algorithmically determine soft selector strings.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2014 Diffeo, Inc.

'''

from __future__ import absolute_import, division, print_function
import argparse
import yakonfig
import dblogger
from operator import itemgetter
from itertools import imap
from collections import defaultdict
import string

import streamcorpus

from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.util import ngrams

from gensim import corpora, models

def make_ngram_corpus(corpus_clean_visibles, n_tokens, punctuation):
    '''
    loads the corpus from the streamcorpus Chunk
    and then tokenizes and then computes n grams
    on each document

    `corpus_path' -- the path to the corpus
    `n' --- the n of the ngrams
    `punctuation' --- if False, punctuation is filtered
    '''
    corpus = list()

    
    if punctuation:
        #tokenize = word_tokenize
        tokenize = lambda s: string.split(s)
        backpage_string = 'backpage.com'
        end_string = 'Poster\'s'

    else:
        ## word tokenizer that removes punctuation
        tokenize = RegexpTokenizer(r'\w+').tokenize
        backpage_string = 'backpage'
        end_string = 'Poster'


    for clean_visible in corpus_clean_visibles:
        ## make tokens
        tokens = tokenize(clean_visible) ## already a unicode string

        # print '  '.join(tokens)
        # print

        ## filter out non backpage pages
        if backpage_string not in tokens: 
            continue

        ## string that signals the beginning of the body
        try:
            idx0 = tokens.index('Reply')
        except:
            continue
        ## string that signals the end of the body
        try:
            idx1 = tokens.index(end_string)
        except:
            continue

        tokens = tokens[idx0:idx1]

        # if 'Yumi' in tokens:
        #     print ' '.join(tokens)
        #     print

        ## make ngrams, attach to make strings
        ngrams_strings = list()
        for ngram_tuple in ngrams(tokens, n_tokens):
            ngrams_strings.append(' '.join(ngram_tuple))
        
        corpus.append(ngrams_strings)

    return corpus


#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('corpus', help='path to the corpus')
#    parser.add_argument('-n', default=6, type=int, help='the n of the ngrams')
#    parser.add_argument('--punctuation', default=False, type=bool, 
#                        help='if False, punctuation is filtered out')
#    args = yakonfig.parse_args(parser, [yakonfig, dblogger])

#    ## make ngram corpus
#    corpus_strings = make_ngram_corpus(args.corpus, args.n, args.punctuation)

def find_soft_selectors(top_results, n_tokens=6, punctuation=False):

    corpus_clean_visibles = map(itemgetter('meta_clean_visible'), 
                                imap(itemgetter(1), top_results))

    corpus_cids = map(itemgetter(0), top_results))

    corpus_strings = make_ngram_corpus(corpus_clean_visibles, n_tokens, punctuation)

    ## make dictionary
    dictionary = corpora.Dictionary(corpus_strings)

    ## make word vectors
    corpus = map(dictionary.doc2bow, corpus_strings)

    ## train tfidf model
    tfidf = models.TfidfModel(corpus)

    ## transform coprus
    corpus_tfidf = tfidf[corpus]

    ## sum up tf-idf across the entire corpus
    corpus_total = defaultdict(int)
    inverted_index = defaultdict(set)
    for doc_idx, doc in enumerate(corpus_tfidf):
        for word_id, score in doc:
            corpus_total[word_id] += score
            inverted_index[word_id].add(corpus_cids[doc_idx])

    ## order the phrases by tf-idf score across the documents
    corpus_ordered = sorted(corpus_total.items(), 
                            key=itemgetter(1),
                            reverse=True,
                            )

    for word_id, score in corpus_ordered:
        return ( score, dictionary[word_id], inverted_index[word_id] )

    #### Do something better with aligning
    ## print in (reverse) score order
    ## the top scoring phrases are at the bottom
    best_score = corpus_total[0][1]
    top_phrases = []
    for word_id, score in corpus_ordered:
        if abs(best_score - score) < 0.01:
            top_phrases.append(( score, dictionary[word_id], inverted_index[word_id] ) )



