'''Algorithmically determine soft selector strings.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2014 Diffeo, Inc.

Soft selectors
==============
.. autofunction:: find_soft_selectors
.. autofunction:: make_ngram_corpus
'''

from __future__ import absolute_import, division, print_function
import argparse
from collections import defaultdict
import logging
from operator import itemgetter
import string

import dblogger
from gensim import corpora
import many_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
import regex as re
import streamcorpus
from streamcorpus_pipeline._clean_html import clean_html
from streamcorpus_pipeline._clean_visible import clean_visible
import yakonfig


logger = logging.getLogger(__name__)
stop_words = many_stop_words.get_stop_words()


def find_soft_selectors(ids_and_clean_visible, start_num_tokens='10',
                        max_num_tokens='20', filter_punctuation='0'):
    '''External interface for dossier.models.soft_selectors.

    This at scans through `num_tokens` values between
    `start_num_tokens` and `max_num_tokens` and calls
    `find_soft_selectors_at_n` looking for results

    All of the params can be passed from URL parameters, in which
    case they can be strings and this function will type cast them
    appropriately.
    '''
    start_num_tokens = int(start_num_tokens)
    max_num_tokens = int(max_num_tokens)
    filter_punctuation = bool(int(filter_punctuation))

    if not ids_and_clean_visible:
        logger.info('find_soft_selectors called with no ids_and_clean_visible')
        return []

    current_results = [] ## results from current n
    previous_results = [] ## previous results from last n
    overall_results = [] ## overall results to return

    for num_tokens in range(start_num_tokens, max_num_tokens + 1):

        ## update this here
        previous_results = current_results

        results_at_n = find_soft_selectors_at_n(
            ids_and_clean_visible, num_tokens, filter_punctuation)
        if len(results_at_n) == 0:
            break

        best_score = results_at_n[0]['score']

        ## i.e. the initial condition is they all have the same score
        idx_at_second = len(results_at_n)

        for idx, result in enumerate(results_at_n):
            if result['score'] < best_score:
                idx_at_second = idx
                break

        current_results = results_at_n[0:idx_at_second]

        if num_tokens == 8:
            for r in results_at_n:
                logger.info('%s --- score: %d' % (r['phrase'], r['score']))

        if previous_results == []:
            logger.info('Previous results are empty. Continuing.')
            continue

        ## now, the main idea is to figure out if any strings from previous
        ## are substrings of those from current
        ## (with the scores fixed at the max for that subphrase).
        ## when they stop being substrings
        ## then those are completed phrases and should be returned as a result

        for prev_result in previous_results:

            is_subbed_and_same_score = False

            for curr_result in current_results:
                if prev_result['phrase'] in curr_result['phrase'] and \
                    prev_result['score'] == curr_result['score'] :

                    is_subbed_and_same_score = True
                    break

            if not is_subbed_and_same_score: ## then it's a honest result
                prev_result['n'] = num_tokens - 1

                overall_results.append(prev_result)

        if len(current_results) == 0:
            ## we got them all
            ## (we still had to collect the previous results)
            ## that's why this break comes after the previous for loop
            break


    ## also add results from current_results at final n
    for result in current_results:
        result['n'] = num_tokens
        overall_results.append(result)

    ## sort by score then by length
    overall_results.sort(key=itemgetter('score', 'n'), reverse=True)

    logger.info('OVERALL RESULTS: %d' % len(overall_results))
    # for idx, result in enumerate(overall_results):
        # logger.info('%d. %s --- score: %f , n = %d, hits=%d' %
            # (idx, result['phrase'], result['score'], result['n'], len(result['hits']))
        # )
    return overall_results

def find_soft_selectors_at_n(ids_and_clean_visible, num_tokens,
                             filter_punctuation):
    corpus_clean_visibles = map(itemgetter(1), ids_and_clean_visible)
    corpus_cids = map(itemgetter(0), ids_and_clean_visible)

    corpus_strings = make_ngram_corpus(
        corpus_clean_visibles, num_tokens, filter_punctuation)


    ## make dictionary
    dictionary = corpora.Dictionary(corpus_strings)

    ## make word vectors
    corpus = map(dictionary.doc2bow, corpus_strings)

    # ## train tfidf model
    # tfidf = models.TfidfModel(corpus)

    # ## transform coprus
    # corpus_tfidf = tfidf[corpus]

    ## ignore tfidf
    corpus_tfidf = corpus

    ## sum up tf-idf across the entire corpus
    corpus_total = defaultdict(int)
    inverted_index = defaultdict(set)
    for doc_idx, doc in enumerate(corpus_tfidf):
        for word_id, score in doc:
            ## uncomment when doing tf idf
            # corpus_total[word_id] += score
            ## this way we only count once per doc
            corpus_total[word_id] += 1

            inverted_index[word_id].add(corpus_cids[doc_idx])

    ## order the phrases by score across the documents
    corpus_ordered = sorted(
        corpus_total.items(), key=itemgetter(1), reverse=True)

    top_phrases = []
    for word_id, score in corpus_ordered:
        top_phrases.append({
            'score': score,
            'phrase': dictionary[word_id],
            'hits': [{'content_id': cid, 'title': None}
                     for cid in inverted_index[word_id]],
        })
    return top_phrases


def make_ngram_corpus(corpus_clean_visibles, num_tokens, filter_punctuation,
                      zoning_rules=False):
    '''takes a list of clean_visible texts, such as from StreamItems or
    FCs, tokenizes all the texts, and constructs n-grams using
    `num_tokens` sized windows.

    ``corpus_clean_visibles`` -- list of unicode strings
    ``num_tokens`` --- the n of the n-grams
    ``filter_punctuation`` --- if True, punctuation is filtered
    '''
    ## TODO: generatlize this zoning code, so that it works on many
    ## sites in the HT domain; consider finishing streamcorpus-zoner
    ## to do this.
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

    corpus = list()
    for clean_vis in corpus_clean_visibles:
        ## crudely skip pages that have "error"
        if re.search(u'error', clean_vis, re.I & re.UNICODE):
            continue

        ## make tokens
        tokens = tokenize(clean_vis) ## already a unicode string

        if zoning_rules:
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

        ## make ngrams, attach to make strings
        ngrams_strings = list()


        for ngram_tuple in ngrams(tokens, num_tokens):
            # ## attempt to remove unwanted phrases
            ## score with many_stop_words and drop bad tuples
            # stop_count = sum([int(bool(tok.lower() in stop_words))
            #                   for tok in ngram_tuple])
            # if stop_count > num_tokens / 1.5:
            #     continue

            ## remove ones with many repeated words
            if len(set(ngram_tuple)) < len(ngram_tuple) / 2:
                continue

            ## this adds ngrams for the current doc
            ngrams_strings.append(' '.join(ngram_tuple))


        ## this adds a list of all the ngrams from the current doc
        ## to the corpus list
        corpus.append(ngrams_strings)
    return corpus


def ids_and_clean_visible_from_streamcorpus_chunk_path(corpus_path):
    '''converts a streamcorpus.Chunk file into the structure that is
    passed by the search engine to find_soft_selectors

    '''
    ch = clean_html(clean_html.default_config)
    cv = clean_visible(clean_visible.default_config)
    ids_and_clean_visible = []
    for si in streamcorpus.Chunk(path=corpus_path):
        if not si.body.clean_visible:
            ## attempt to make clean_visible
            if not si.body.raw:
                logger.critical('no raw content, so skipping: %r', si.abs_url)
                continue
            abs_url = si.abs_url
            si = ch(si, {})
            if not si:
                logger.critical(
                    'failed to make clean_html, so skipping: %r', abs_url)
                continue
            si = cv(si, {})
            if not si or not si.body.clean_visible:
                logger.critical(
                    'failed to make clean_visible, so skipping: %r', abs_url)
                continue
        rec = (si.stream_id, si.body.clean_visible.decode('utf8'), {})
        ids_and_clean_visible.append(rec)
    return ids_and_clean_visible


def main():
    parser = argparse.ArgumentParser(
        'command line tool for debugging and development')
    parser.add_argument('corpus', help='path to a streamcorpus.Chunk file')
    parser.add_argument('-n', '--num-tokens', default=6, type=int,
                        help='the n of the ngrams; used as start_num_tokens '
                             'for scanning')
    parser.add_argument('--max-num-tokens', default=40, type=int,
                        help='maximum number of `n` in n-grams for scanning')
    parser.add_argument('--peak-score-delta', default=0.01, type=float,
                        help='delta in score values required between first '
                             'and second result to stop  scanning')
    parser.add_argument('--scan-window-size', default=False,
                        action='store_true',
                        help='if set, scans from the value of -n until it '
                             'finds a strongly peaked top value')
    parser.add_argument('--filter-punctuation', default=False,
                        action='store_true',
                        help='filter out punctuation; default is to not '
                             'filter punctuation')
    parser.add_argument('--show-ids', default=False, action='store_true',
                        help='show identifiers in diagnostic output')
    args = yakonfig.parse_args(parser, [yakonfig, dblogger])

    ## TODO: if we start needing to load FC chunk files (instead of SI
    ## chunk files), this might need to be told which kind of chunk it
    ## is loading, and we'll need a second function along the lines of
    ## ids_and_clean_visible_from_streamcorpus_chunk_path

    ## mimic the in-process interface:
    ids_and_clean_visible = ids_and_clean_visible_from_streamcorpus_chunk_path(
        args.corpus)
    logger.info('gathered %d texts', len(ids_and_clean_visible))

    def format_result(result):
        score, soft_selector_phrase, matching_texts = result
        return '%.6f\t%d texts say:\t%s\t%s' % \
            (score, len(matching_texts), soft_selector_phrase.encode('utf8'),
             args.show_ids and repr(matching_texts) or '')

    if args.scan_window_size:
        best = find_soft_selectors(
            ids_and_clean_visible,
            start_num_tokens=args.num_tokens,
            max_num_tokens=args.max_num_tokens,
            filtered_punctuation=args.filter_punctuation)
        if not best:
            print('failed to find a best result!')
        else:
            print('found a best result:')
            print('\n'.join(map(format_result, best)))

    else:
        results = find_soft_selectors_at_n(
            ids_and_clean_visible, args.num_tokens, args.filter_punctuation)
        print('\n'.join(map(format_result, results)))


if __name__ == '__main__':
    main()
