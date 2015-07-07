from __future__ import absolute_import, division, print_function

from collections import defaultdict
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import hashlib
from itertools import imap
import logging
from operator import itemgetter
import os.path as path
import urllib

from bs4 import BeautifulSoup
import bottle
import cbor
import json
from streamcorpus_pipeline import cleanse

import dblogger
from dossier.fc import StringCounter, FeatureCollection
from dossier.models import etl
from dossier.models.extractor import extract
from dossier.models.folder import Folders
from dossier.models.openquery.fetcher import Fetcher
from dossier.models.pairwise import negative_subfolder_ids
from dossier.models.report import ReportGenerator
from dossier.models.subtopic import subtopics, subtopic_type, typed_subtopic_data
from dossier.models.web.config import Config
import dossier.web.routes as routes
from dossier.web.util import fc_to_json
import kvlayer
import rejester
from rejester._task_master import \
    AVAILABLE, BLOCKED, PENDING, FINISHED
import yakonfig
import operator


app = bottle.Bottle()
logger = logging.getLogger(__name__)
web_static_path = path.join(path.split(__file__)[0], 'static')
bottle.TEMPLATE_PATH.insert(0, path.join(web_static_path, 'tpl'))


@app.get('/SortingQueue')
def example_sortingqueue():
    return bottle.template('example-sortingqueue.html')


@app.get('/SortingDesk')
def example_sortingdesk():
    return bottle.template('example-sortingdesk.html')


@app.get('/static/<name:path>')
def v1_static(name):
    return bottle.static_file(name, root=web_static_path)


@app.get('/dossier/v1/folder/<fid>/report')
def v1_folder_report(request, response, kvlclient, store, fid):
    response.headers['Content-Disposition'] = \
        'attachment; filename="report-%s.xlsx"' % fid
    response.headers['Content-Type'] = \
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    folders = new_folders(kvlclient, request)
    gen = ReportGenerator(store, folders, urllib.unquote(fid))
    body = StringIO()
    gen.run(body)
    return body.getvalue()


@app.get('/dossier/v1/folder/<fid>/subfolder/<sid>/extract')
def v1_folder_extract_get(request, response, kvlclient, store, fid, sid):
    conf = yakonfig.get_global_config('rejester')
    tm = rejester.build_task_master(conf)
    key = cbor.dumps((fid, sid))
    wu_status = tm.get_work_unit_status('ingest', key)
    status = wu_status['status']
    if status in (AVAILABLE, BLOCKED, PENDING):
        return {'state': 'pending'}
    elif status in (FINISHED,):
        return {'state': 'done'}
    else:
        return {'state': 'failed'}


@app.post('/dossier/v1/folder/<fid>/subfolder/<sid>/extract')
def v1_folder_extract_post(fid, sid):
    logger.info('launching async work unit for %r', (fid, sid))
    conf = yakonfig.get_global_config('rejester')
    tm = rejester.build_task_master(conf)
    tm.add_work_units('ingest', [(cbor.dumps((fid, sid)), {})])


def rejester_run_extract(work_unit):
    if 'config' not in work_unit.spec:
        raise rejester.exceptions.ProgrammerError(
            'could not run profile import without global config')

    web_conf = Config()
    unitconf = work_unit.spec['config']
    with yakonfig.defaulted_config([rejester, kvlayer, dblogger, web_conf],
                                   config=unitconf):
        fid, sid = cbor.loads(work_unit.key)
        tfidf = web_conf.tfidf
        folders = Folders(web_conf.kvlclient)
        fetcher = Fetcher()

        # This is where you can enable the keyword extractor.
        # Comment out the next two lines (`get_subfolder_queries`)
        # and uncomment the following two lines (`extract_keyword_queries`).

        # queries = get_subfolder_queries(
        #     web_conf.store, web_conf.label_store, folders, fid, sid)

        queries = extract_keyword_queries(
            web_conf.store, web_conf.label_store, folders, fid, sid)

        link2queries = defaultdict(set)

        logger.info('searching google for: %r', queries)
        for q in queries:
            for result in web_conf.google.web_search_with_paging(q, limit=10):
                map(link2queries[result['link']].add, cleanse(q.decode('utf8')).split())

        result = None

        logger.info('got %d URLs from %d queries', len(link2queries), len(queries))

        for link, queries in link2queries.items():
            si = fetcher.get(link)
            cid_url = hashlib.md5(str(link)).hexdigest()
            cid = etl.interface.mk_content_id(cid_url)

            # hack alert!
            # We currently use FCs to store subtopic text data, which
            # means we cannot overwrite existing FCs with reckless
            # abandon. So we adopt a heuristic: check if an FC already
            # exists, and if it does, check if it is being used to store
            # user data. If so, don't overwrite it and move on.
            fc = web_conf.store.get(cid)
            if fc is not None and any(k.startswith('subtopic|')
                                      for k in fc.iterkeys()):
                logger.info('skipping ingest for %r (abs url: %r) because '
                            'an FC with user data already exists.',
                            cid, link)
                continue

            other_features = {
                u'keywords': StringCounter(list(queries)),
            }

            try:
                fc = create_fc_from_html(
                    link, si.body.raw,
                    encoding=si.body.encoding or 'utf-8', tfidf=tfidf,
                    other_features=other_features,
                )
                logger.info('created FC for %r (abs url: %r)',
                            cid, link)
                web_conf.store.put([(cid, fc)])
            except Exception:
                logger.info('failed ingest on %r (abs url: %r)',
                            cid, link, exc_info=True)


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


def extract_keyword_queries(store, label_store, folders, fid, sid):
    ids = map(itemgetter(0), folders.items(fid, sid))
    positive_fcs = map(itemgetter(1), store.get_many(ids))
    negative_ids = imap(itemgetter(0),
                        negative_subfolder_ids(label_store, folders, fid, sid))
    negative_fcs = map(itemgetter(1), store.get_many(negative_ids))

    ## GPE seems like a good feature to use
    pos_words, neg_words = extract(positive_fcs, negative_fcs,
                                   features=['GPE'])

    ## so my assumption is that fid is the `name' of the entity in question
    ## or, more generally, the standard query to be executed
    ## i know this is slightly different than what andrew assumed
    ## but i think it makes sense for the annotated version of this task

    query_names = fid.split('_')
    ## quotes added so that google treats the name as one token
    original_query = '\"' + ' '.join(query_names) + '\"'
    logger.info('the original query was %s', original_query)

    ## return five queries with the original_query name,
    ## 0. the original name --- the pairwise model will eliminate if bad
    ##    but it seems like it's a mistake to omit
    ## 1. plus the most predictive keyword
    ## 2. minus the least predictive keyword
    ## 3. minus the most predictive keyword for the negative class
    ## 4. plus the least predictive keyword for the negative class

    ## additionally, if any of these words are the name, we skip to the
    ## next keyword in the list

    queries = []

    ## 0. original name
    queries.append(original_query)

    ## 1. plus the most predictive keyword
    query_plus_pred = original_query + ' ' + \
                                name_filter(pos_words, query_names)
    logger.info('query 1: + most predictive %s', query_plus_pred)
    queries.append(query_plus_pred)

    ## 2. minus the least predictive keyword
    query_min_least = original_query + ' -' + \
                                name_filter(reversed(pos_words), query_names)
    logger.info('query 2: - least predictive %s', query_min_least)
    queries.append(query_min_least)

    ## 3. minus the most predictive keyword for the negative class
    query_min_most_neg = original_query + ' -' + \
                                name_filter(neg_words, query_names)
    logger.info('query 3: - most predictive for neg %s', query_min_most_neg)
    queries.append(query_min_most_neg)

    ## 4. plus the least predictive keyword for the negative class
    query_plus_least_neg = original_query + ' ' + \
                                name_filter(reversed(neg_words), query_names)
    logger.info('query 4: + least predictive for neg %s', query_plus_least_neg)
    queries.append(query_plus_least_neg)



    # logger.info('length %d', len(positive_fcs))

    # for fc in positive_fcs:
    #     logger.info('pos fc %r', fc['title'])


    # logger.info('pos fc %r', positive_fcs[3]['GPE'])

    # logger.info('pos fc %r', positive_fcs[3].keys())
    # logger.info('pos fc %r', positive_fcs[3]['PERSON'])

    # logger.info('positive keywords: %r', pos_words)
    # logger.info('negative keywords: %r', neg_words)

    # logger.info('most positive keyword: %r', pos_words[0])


    return queries

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




def OLD_v1_folder_extract_post(request, response, kvlclient, store,
                               label_store, fid, sid):
    # TODO: this block of code needs to move into a `coordinate` run
    # function and then get extended to:

    # 1) include querying the Google Custom Search API by copying/porting the
    # web2dehc/google.py

    # 2) fetch the URLs, using code ported/copied out of
    # web2dehc/fetcher.py

    # 3) ingested into dossier.store using streamcorpus_pipeline and
    # dossier.models.etl.interface.to_dossier_store

    folders = new_folders(kvlclient, request)
    ids = map(itemgetter(0), folders.items(fid, sid))
    positive_fcs = map(itemgetter(1), store.get_many(ids))
    negative_ids = imap(itemgetter(0),
                        negative_subfolder_ids(label_store, folders, fid, sid))
    negative_fcs = map(itemgetter(1), store.get_many(negative_ids))
    pos_words, neg_words = extract(positive_fcs, negative_fcs,
                                   features=['bowNP_sip'])
    return {'postive': dict(pos_words),
            'negative': dict(neg_words)}


@app.put('/dossier/v1/feature-collection/<cid>', json=True)
def v1_fc_put(request, response, store, tfidf, cid):
    '''Store a single feature collection.

    The route for this endpoint is:
    ``PUT /dossier/v1/feature-collections/<content_id>``.

    ``content_id`` is the id to associate with the given feature
    collection. The feature collection should be in the request
    body serialized as JSON.

    Alternatively, if the request's ``Content-type`` is
    ``text/html``, then a feature collection is generated from the
    HTML. The generated feature collection is then returned as a
    JSON payload.

    This endpoint returns status ``201`` upon successful
    storage otherwise. An existing feature collection with id
    ``content_id`` is overwritten.
    '''
    tfidf = tfidf or None
    if request.headers.get('content-type', '').startswith('text/html'):
        url = urllib.unquote(cid.split('|', 1)[1])
        fc = create_fc_from_html(url, request.body.read(), tfidf=tfidf)
        logger.info('created FC for %r', cid)
        store.put([(cid, fc)])
        return fc_to_json(fc)
    else:
        fc = FeatureCollection.from_dict(json.load(request.body))
        keywords = set()
        for subid in fc:
            if subid.startswith('subtopic'):
                ty = subtopic_type(subid)
                if ty in ('text', 'manual'):
                    # get the user selected string
                    data = typed_subtopic_data(fc, subid)
                    map(keywords.add, cleanse(data).split())

        fc[u'keywords'] = StringCounter(keywords)
        store.put([(cid, fc)])
        response.status = 201

        #return routes.v1_fc_put(request, response, lambda x: x, store, cid)


def create_fc_from_html(url, html, encoding='utf-8', tfidf=None, other_features=None):
    soup = BeautifulSoup(unicode(html, encoding))
    title = soup_get(soup, 'title', lambda v: v.get_text())
    body = soup_get(soup, 'body', lambda v: v.prettify())
    if other_features is None:
        other_features = {}
    other_features.update({
        u'title': StringCounter([title]),
        u'titleBow': StringCounter(title.split()),
    })
    fc = etl.html_to_fc(body, url=url, other_features=other_features)
    if fc is None:
        return None
    if tfidf is not None:
        etl.add_sip_to_fc(fc, tfidf)
    return fc


def soup_get(soup, sel, cont):
    v = soup.find(sel)
    if v is None:
        return u''
    else:
        return cont(v)


def new_folders(kvlclient, request):
    conf = {}
    if 'annotator_id' in request.query:
        conf['owner'] = request.query['annotator_id']
    return Folders(kvlclient, **conf)
