from __future__ import absolute_import, division, print_function

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

import dblogger
from dossier.fc import StringCounter
from dossier.models import etl
from dossier.models.extractor import extract
from dossier.models.folder import Folders
from dossier.models.openquery.fetcher import Fetcher
from dossier.models.pairwise import negative_subfolder_ids
from dossier.models.report import ReportGenerator
from dossier.models.subtopic import subtopics
from dossier.models.web.config import Config
import dossier.web.routes as routes
from dossier.web.util import fc_to_json
import kvlayer
import rejester
from rejester._task_master import \
    AVAILABLE, BLOCKED, PENDING, FINISHED
import yakonfig


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
        queries = get_subfolder_queries(
            web_conf.store, web_conf.label_store, folders, fid, sid)
        # queries = extract_keyword_queries(
        #     web_conf.store, web_conf.label_store, folders, fid, sid)
        logger.info('searching google for: %r', queries)
        for q in queries:
            for result in web_conf.google.web_search_with_paging(q, limit=10):
                si = fetcher.get(result['link'])
                cid_url = hashlib.md5(str(result['link'])).hexdigest()
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
                                cid, result['link'])
                    continue

                try:
                    fc = create_fc_from_html(
                        result['link'], si.body.raw,
                        encoding=si.body.encoding or 'utf-8', tfidf=tfidf)
                    logger.info('created FC for %r (abs url: %r)',
                                cid, result['link'])
                    web_conf.store.put([(cid, fc)])
                except Exception:
                    logger.info('failed ingest on %r (abs url: %r)',
                                cid, result['link'], exc_info=True)


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
    pos_words, neg_words = extract(positive_fcs, negative_fcs,
                                   features=['bowNP_sip'])
    # Stupidly limit it to 5 keywords for now, so that we don't spawn a huge
    # number of Google searches.
    return (pos_words.keys() + neg_words.keys())[0:5]


# I don't know how to make this code do useful things. The extractor seems
# like a good idea, but its quality is limited to the quality of our features.
# The best feature we have is bowNP_sip, which doesn't seem to be very good.
#
# Instead, I've opted to do searching based on the names of items assigned
# by the user.
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
        return routes.v1_fc_put(request, response, lambda x: x, store, cid)


def create_fc_from_html(url, html, encoding='utf-8', tfidf=None):
    soup = BeautifulSoup(unicode(html, encoding))
    title = soup_get(soup, 'title', lambda v: v.get_text())
    body = soup_get(soup, 'body', lambda v: v.prettify())
    fc = etl.html_to_fc(body, url=url, other_features={
        u'title': StringCounter([title]),
        u'titleBow': StringCounter(title.split()),
    })
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
