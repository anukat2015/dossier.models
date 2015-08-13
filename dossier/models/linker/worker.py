'''coordinate worker function for executing query generation
from a set of dossiers for use with open query.

.. autofunction:: worker
.. autofunction:: traverse_extract_fetch
.. autofunction:: get_subfolder_queries
.. autofunction:: extract_keyword_queries
.. autofunction:: name_filter

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.
'''
from __future__ import absolute_import, division, print_function
import cbor
import hashlib
from itertools import imap, islice
import json
from operator import itemgetter
import logging

from dossier.fc import StringCounter, FeatureCollection
import dblogger
import kvlayer
import coordinate
from streamcorpus_pipeline import cleanse
import yakonfig

from dossier.models.subtopic import subtopics
from dossier.models.openquery.fetcher import Fetcher
from dossier.models.folder import Folders
from dossier.models.pairwise import negative_subfolder_ids
from dossier.models import etl
from dossier.models.web.config import Config
from .model import extract

logger = logging.getLogger(__name__)

def worker(work_unit):
    '''Expects a WorkUnit from coordinated, obtains a config, and runs
    traverse_extract_fetch

    '''
    if 'config' not in work_unit.spec:
        raise coordinate.exceptions.ProgrammerError(
            'could not run extraction without global config')

    web_conf = Config()
    unitconf = work_unit.spec['config']
    #logger.info(unitconf)
    with yakonfig.defaulted_config([coordinate, kvlayer, dblogger, web_conf],
                                   config=unitconf):
        traverse_extract_fetch(web_conf, work_unit.key)


def traverse_extract_fetch(config, wukey, stop_after_extraction=False):
    '''Given a config and a
    `wukey=cbor.dumps((folder_name,subfolder_name))`, traverse the
    folders to generate queries, issue them to Google, fetch the
    results, and ingest them.

    '''

    config.kvlclient.setup_namespace({'openquery': (str,)})
    try:
        data = list(config.kvlclient.get('openquery', (wukey,)))
        if data:
            if data[0][1]:
                logger.info('found existing query results: %r', data)
                return
            else:
                config.kvlclient.delete('openquery', (wukey,))
    except:
        logger.error('failed to get data from existing table', exc_info=True)

    fid, sid = cbor.loads(wukey)
    tfidf = config.tfidf
    folders = Folders(config.kvlclient)
    fetcher = Fetcher()

    ## To disable the keyword extractor model, you can uncomment out
    ## the next three lines (`get_subfolder_queries`) and comment out
    ## the following two lines (`extract_keyword_queries`).
    #keyword_feature_keys = []
    #queries = get_subfolder_queries(
    #    config.store, config.label_store, folders, fid, sid)

    queries, keyword_feature_keys, has_observations = extract_keyword_queries(
        config.store, config.label_store, folders, fid, sid)

    logger.info('Model found %d queries: %r', len(queries), queries)

    if stop_after_extraction:
        return

    keywords = set()
    for key in keyword_feature_keys:
        ckey = cleanse(key.decode('utf8'))
        keywords.add(ckey)
        for part in ckey.split():
            keywords.add(part)

    #link2queries = defaultdict(set)
    links = set()
    logger.info('searching google for: %r', queries)
    for q in queries:
        for result in config.google.web_search_with_paging(q, limit=10):
            links.add(result['link'])
            #map(link2queries[result['link']].add, cleanse(q.decode('utf8')).split())
            logger.info('discovered %r', result['link'])

    result = None

    #logger.info('got %d URLs from %d queries', len(link2queries), len(queries))
    logger.info('got %d URLs from %d queries', len(links), len(queries))

    # content_ids gets modified within the 'callback' closure
    content_ids = []
    #for link, queries in link2queries.items():

    def callback(si, link):
        if si is None: return
        cid_url = hashlib.md5(str(link)).hexdigest()
        cid = etl.interface.mk_content_id(cid_url)
        content_ids.append(cid)

        # hack alert!
        # We currently use FCs to store subtopic text data, which
        # means we cannot overwrite existing FCs with reckless
        # abandon. So we adopt a heuristic: check if an FC already
        # exists, and if it does, check if it is being used to store
        # user data. If so, don't overwrite it and move on.
        fc = config.store.get(cid)
        if fc is not None and any(k.startswith('subtopic|')
                                  for k in fc.iterkeys()):
            logger.info('skipping ingest for %r (abs url: %r) because '
                        'an FC with user data already exists.',
                        cid, link)
            return

        other_features = {
            u'keywords': StringCounter(keywords), #list(queries)),
        }

        try:
            fc = etl.create_fc_from_html(
                link, si.body.raw,
                encoding=si.body.encoding or 'utf-8', tfidf=tfidf,
                other_features=other_features,
            )
            if not fc:
                logger.info('failed to get an FC, moving on')
                return
            logger.info('created FC for %r (abs url: %r)',
                        cid, link)
            config.store.put([(cid, fc)])
        except Exception:
            logger.info('trapped ingest failure on %r (abs url: %r)',
                        cid, link, exc_info=True)

    logger.info('FETCHING using ASYNC')
    fetcher.get_async(islice(links, None), callback)

    data = json.dumps({'content_ids': content_ids})
    logger.info('saving %d content_ids in %d bytes on wukey %r',
                len(content_ids), len(data), wukey)
    config.kvlclient.put('openquery', ((wukey,), data))
    logger.info('done saving for %r', wukey)

def get_subfolder_queries(store, label_store, folders, fid, sid):
    '''Returns [unicode].

    This returns a list of queries that can be passed on to "other"
    search engines. The list of queries is derived from the subfolder
    identified by ``fid/sid``.
    '''
    queries = []

    for cid, subid, url, stype, data in subtopics(store, folders, fid, sid):
        if stype in ('text', 'manual'):
            queries.append(data)
    return queries


def extract_keyword_queries(store, label_store, folders, fid, sid, include_original=False):
    '''Transforms a folder structure into positive and negative examples
    to feed to ``linker.model.extract``.  This transforms SortingDesk's
    foldering structure into *supervision* data for the extractor.

    This works best if folder name (``fid``) is the ``name`` of an entity
    in question or, more generally, a query that a user might have
    issued to a search engine.  In particular, this approach makes
    sense for the annotated version of this task, which is what
    SortingDesk enables.

    This returns five queries with the original_query name:

      0. if `include_original`, then original name; the model will
         eliminate if bad but it seems like it's a mistake to omit

      1. plus the most predictive keyword

      2. minus the least predictive keyword

      3. minus the most predictive keyword for the negative class

      4. plus the least predictive keyword for the negative class

    Additionally, if any of these words are the name, we skip to the
    next keyword in the list.

    Returns a three tuple of ([unicode], [unicode], bool) where the
    first list is query strings to send to a search engine, and the
    second list is feature strings to put in a StringCounter.

    '''
    keyword_feature_keys = []

    query_names = fid.split('_')
    ## quotes added so that google treats the name as one token
    name1 = ' '.join(query_names)
    #keyword_feature_keys.append(name1)
    original_query = '\"' + name1  + '\"'
    logger.info('the original query was %s', original_query)

    queries = []

    ## 0. original name
    if include_original:
        logger.info('query 0: including the original: %r', original_query)
        queries.append(original_query)

    if sid:
        name2 = ' '.join(sid.split('_'))
        keyword_feature_keys.append(name2)
        queries.append( '\"' + name2 + '\"' )

    ## generate positive and negative examples by traversing folders
    try:
        ids = map(itemgetter(0), folders.items(fid, sid))
    except KeyError:
        logger.info('Folder traversal failed to find ids, so no '
                    'training data; giving up on model-based queries')
        # third return value of `False` means no observations
        return queries, keyword_feature_keys, False

    positive_fcs = map(itemgetter(1), store.get_many(ids))
    negative_ids = imap(itemgetter(0),
                        negative_subfolder_ids(label_store, folders, fid, sid))
    negative_fcs = map(itemgetter(1), store.get_many(negative_ids))

    ## These features were selected by manual inspection of current
    ## FOSS NER output.
    pos_words, neg_words = extract(positive_fcs, negative_fcs,
                                   features=['GPE', 'PERSON', 'ORGANIZATION'])

    ## 1. plus the most predictive keyword
    query_plus_pred = original_query + ' ' + \
                                name_filter(pos_words, query_names)
    logger.info('query 1: + most predictive: %r', query_plus_pred)
    queries.append(query_plus_pred)

    ## 2. minus the least predictive keyword
    query_min_least = original_query + ' -' + \
                                name_filter(reversed(pos_words), query_names)
    logger.info('query 2: - least predictive: %r', query_min_least)
    queries.append(query_min_least)

    ## 3. minus the most predictive keyword for the negative class
    query_min_most_neg = original_query + ' -' + \
                                name_filter(neg_words, query_names)
    logger.info('query 3: - most predictive for neg: %r', query_min_most_neg)
    queries.append(query_min_most_neg)

    ## 4. plus the least predictive keyword for the negative class
    query_plus_least_neg = original_query + ' ' + \
                                name_filter(reversed(neg_words), query_names)
    logger.info('query 4: + least predictive for neg: %r', query_plus_least_neg)
    queries.append(query_plus_least_neg)

    ## for debugging
    # logger.info('length %d', len(positive_fcs))

    # for fc in positive_fcs:
    #     logger.info('pos fc %r', fc['title'])


    # logger.info('pos fc %r', positive_fcs[3]['GPE'])

    # logger.info('pos fc %r', positive_fcs[3].keys())
    # logger.info('pos fc %r', positive_fcs[3]['PERSON'])

    # logger.info('positive keywords: %r', pos_words)
    # logger.info('negative keywords: %r', neg_words)

    # logger.info('most positive keyword: %r', pos_words[0])

    return queries, keyword_feature_keys, True


def name_filter(keywords, names):
    '''
    Returns the first keyword from the list, unless
    that keyword is one of the names in names, in which case
    it continues to the next keyword.

    Since keywords consists of tuples, it just returns the first
    element of the tuple, the keyword. It also adds double
    quotes around the keywords, as is appropriate for google queries.

    Input Arguments:
    keywords -- a list of (keyword, strength) tuples
    names -- a list of names to be skipped
    '''
    name_set = set(name.lower() for name in names)

    for key_tuple in keywords:
        if not key_tuple[0] in name_set:
            return '\"' + key_tuple[0] +'\"'

    ## returns empty string if we run out, which we shouldn't
    return ''

