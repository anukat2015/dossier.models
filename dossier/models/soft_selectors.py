'''Algorithmically determine soft selector strings.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2014 Diffeo, Inc.

'''

from __future__ import absolute_import, division, print_function
import argparse
from collections import defaultdict
from itertools import imap
import logging
from operator import itemgetter
import string

import dblogger
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.util import ngrams
import streamcorpus
import yakonfig


logger = logging.getLogger(__name__)



def find_soft_selectors(ids_and_clean_visible, start_num_tokens='6', max_num_tokens='40', 
                        peak_score_delta='0.01',
                        filter_punctuation='0', **kwargs):
    '''external interface for dossier.models.soft_selectors.  This at
    scans through `num_tokens` values between `start_num_tokens` and
    `max_num_tokens` and calls `find_soft_selectors_at_n` looking for
    results in which the top result is more than `peak_score_delta`
    above the second result.


    All of the params can be passed from URL parameters, in which case
    they can be strings and this function will type cast them
    appropriately.

    '''
    if isinstance(start_num_tokens, basestring):
        start_num_tokens = int(start_num_tokens)
    if isinstance(max_num_tokens, basestring):
        max_num_tokens = int(max_num_tokens)
    if isinstance(peak_score_delta, basestring):
        peak_score_delta = float(peak_score_delta)
    if isinstance(filter_punctuation, basestring):
        filter_punctuation = bool(int(filter_punctuation))

    for num_tokens in range(start_num_tokens, max_num_tokens + 1):
        results = find_soft_selectors_at_n(ids_and_clean_visible, num_tokens, filter_punctuation)
        best_score = results[0][0]
        second_best_score = results[1][0]
        logger.info('num_tokens=%d, best_score(%f) - second_best_score(%f)=%f' % 
                    (num_tokens, best_score, second_best_score, (best_score - second_best_score)))
        if second_best_score + peak_score_delta < best_score:
            return results[0]

    ## if we do not find one, return None, so {'suggestions': null}
    return None


def make_ngram_corpus(corpus_clean_visibles, num_tokens, filter_punctuation):
    '''takes a list of clean_visible texts, such as from StreamItems or
    FCs, tokenizes all the texts, and constructs n-grams using
    `num_tokens` sized windows.

    `corpus_clean_visibles' -- list of unicode strings
    `num_tokens' --- the n of the n-grams
    `filter_punctuation' --- if True, punctuation is filtered

    '''
    corpus = list()
    
    if filter_punctuation:
        ## word tokenizer that removes punctuation
        tokenize = RegexpTokenizer(r'\w+').tokenize
        backpage_string = 'backpage'
        end_string = 'Poster'

    else:
        #tokenize = word_tokenize
        tokenize = lambda s: string.split(s)
        backpage_string = 'backpage.com'
        end_string = 'Poster\'s'


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
        for ngram_tuple in ngrams(tokens, num_tokens):
            ngrams_strings.append(' '.join(ngram_tuple))
        
        corpus.append(ngrams_strings)

    return corpus



def find_soft_selectors_at_n(ids_and_clean_visible, num_tokens, filter_punctuation):

    corpus_clean_visibles = map(itemgetter(1), ids_and_clean_visible)
    corpus_cids = map(itemgetter(0), ids_and_clean_visible)

    corpus_strings = make_ngram_corpus(corpus_clean_visibles, num_tokens, filter_punctuation)

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

    top_phrases = []
    for word_id, score in corpus_ordered:
        top_phrases.append(( score, dictionary[word_id], inverted_index[word_id] ) )

    return top_phrases


def ids_and_clean_visible_from_streamcorpus_chunk_path(corpus_path):
    '''converts a streamcorpus.Chunk file into the structure that is
    passed by the search engine to find_soft_selectors

    '''
    return [(si.stream_id, si.body.clean_visible.decode('utf8'), {})
            for si in streamcorpus.Chunk(path=corpus_path)]


def main():
    parser = argparse.ArgumentParser('command line tool for debugging and development')
    parser.add_argument('corpus', help='path to a streamcorpus.Chunk file')
    parser.add_argument('-n', '--num-tokens', default=6, type=int, 
                        help='the n of the ngrams; used as start_num_tokens for scanning')
    parser.add_argument('--max-num-tokens', default=40, type=int, 
                        help='maximum number of `n` in n-grams for scanning')
    parser.add_argument('--peak-score-delta', default=0.01, type=float, 
                        help='delta in score values required between first and second'
                        ' result to stop  scanning')
    parser.add_argument('--scan-window-size', default=False, action='store_true', 
                        help='if set, scans from the value of -n until it finds '
                        'a strongly peaked top value')
    parser.add_argument('--filter-punctuation', default=False, action='store_true', 
                        help='filter out punctuation; default is to not filter punctuation')
    parser.add_argument('--show-ids', default=False, action='store_true', 
                        help='show identifiers in diagnostic output')
    args = yakonfig.parse_args(parser, [yakonfig, dblogger])

    ## TODO: if we start needing to load FC chunk files (instead of SI
    ## chunk files), this might need to be told which kind of chunk it
    ## is loading, and we'll need a second function along the lines of
    ## ids_and_clean_visible_from_streamcorpus_chunk_path

    ## mimic the in-process interface:
    ids_and_clean_visible = ids_and_clean_visible_from_streamcorpus_chunk_path(args.corpus)

    def format_result(result):
        score, soft_selector_phrase, matching_texts = result
        return '%.6f\t%d texts say:\t%s\t%s' % \
            (score, len(matching_texts), soft_selector_phrase, 
             args.show_ids and repr(matching_texts) or '')


    if args.scan_window_size:
        best = find_soft_selectors(ids_and_clean_visible, 
                                   start_num_tokens=args.num_tokens, 
                                   max_num_tokens=args.max_num_tokens, 
                                   filtered_punctuation=args.filter_punctuation)
        if not best:
            print('failed to find a best result!')
        else:
            print('found a best result:')
            print(format_result(best))

    else:
        results = find_soft_selectors_at_n(ids_and_clean_visible, args.num_tokens, 
                                           args.filter_punctuation)

        print('\n'.join(map(format_result, results)))


if __name__ == '__main__':
    main()
