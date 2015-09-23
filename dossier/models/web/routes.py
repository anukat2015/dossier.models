'''web service endpoints for supporting SortingDesk

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.
'''
from __future__ import absolute_import, division, print_function

from collections import defaultdict
from operator import itemgetter
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import logging
import os.path as path
import operator
import urllib

import bottle
import cbor
import json
from streamcorpus_pipeline import cleanse
from streamcorpus_pipeline._clean_html import uniform_html
from streamcorpus_pipeline.offsets import char_offsets_to_xpaths
import regex as re

import dblogger
from dossier.fc import StringCounter, FeatureCollection
from dossier.models import etl
from dossier.models.folder import Folders
from dossier.models.report import ReportGenerator
from dossier.models.subtopic import subtopics, subtopic_type, typed_subtopic_data
from dossier.models.web.config import Config
import dossier.web.routes as routes
from dossier.web.util import fc_to_json
import kvlayer
import coordinate
from coordinate.constants import \
    AVAILABLE, BLOCKED, PENDING, FINISHED, FAILED
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


DRAGNET_KEY = 'only-one-dragnet'

@app.post('/dossier/v1/dragnet')
def v1_dragnet():
    status = dragnet_status()
    if not status or status in (FINISHED, FAILED):
        logger.info('launching dragnet async work unit')
        conf = yakonfig.get_global_config('coordinate')
        tm = coordinate.TaskMaster(conf)
        tm.add_work_units('dragnet', [(DRAGNET_KEY, {})])
        return {'state': 'submitted'}
    else:
        return {'state': 'pending'}

@app.get('/dossier/v1/dragnet')
def v1_dragnet(kvlclient):
    status = dragnet_status()
    if status == PENDING:
        return {'state': 'pending'}
    else:
        kvlclient.setup_namespace({'dragnet': (str,)})
        data = list(kvlclient.get('dragnet', ('dragnet',)))
        if data[0][1]:
            #logger.info(data)
            return json.loads(data[0][1])
        else:
            return {'state': 'failed'}

def dragnet_status():
    conf = yakonfig.get_global_config('coordinate')
    tm = coordinate.TaskMaster(conf)
    wu_status = tm.get_work_unit_status('dragnet', DRAGNET_KEY)
    if not wu_status: return None
    status = wu_status['status']
    return status


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
    conf = yakonfig.get_global_config('coordinate')
    tm = coordinate.TaskMaster(conf)
    key = cbor.dumps((fid, sid))
    wu_status = tm.get_work_unit_status('ingest', key)
    status = wu_status['status']
    if status in (AVAILABLE, BLOCKED, PENDING):
        return {'state': 'pending'}
    elif status in (FINISHED,):
        kvlclient.setup_namespace({'openquery': (str,)})
        data = None
        try:
            data = list(kvlclient.get('openquery', (key,)))
            assert len(data) == 1, data
            logger.info('got data of len 1: %r', data)
            assert data[0], data
            assert data[0][1], data
            data = data[0][1]
            data = json.loads(data)
            data['state'] = 'done'
            return data
        except:
            logger.info('kvlclient: %r', kvlclient)
            logger.error('Failed to get openquery data: %r', data, exc_info=True)
            return {'state': 'failed'}

    else:
        return {'state': 'failed'}


@app.post('/dossier/v1/folder/<fid>/subfolder/<sid>/extract')
def v1_folder_extract_post(fid, sid):
    conf = yakonfig.get_global_config('coordinate')
    tm = coordinate.TaskMaster(conf)
    key = cbor.dumps((fid, sid))
    wu_status = tm.get_work_unit_status('ingest', key)
    if wu_status and wu_status['status'] in (AVAILABLE, BLOCKED, PENDING):
        return {'state': 'pending'}
    else:
        logger.info('launching async work unit for %r', (fid, sid))
        conf = yakonfig.get_global_config('coordinate')
        tm = coordinate.TaskMaster(conf)
        tm.add_work_units('ingest', [(cbor.dumps((fid, sid)), {})])
        return {'state': 'submitted'}


@app.put('/dossier/v1/feature-collection/<cid>', json=True)
def v1_fc_put(request, response, store, kvlclient, tfidf, cid):
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
        fc = etl.create_fc_from_html(url, request.body.read(), tfidf=tfidf)
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
                    keywords.add(cleanse(data))

        folders = Folders(kvlclient)
        for fid, sid in folders.parent_subfolders(cid):
            if not isinstance(fid, unicode):
                fid = fid.decode('utf8')
            if not isinstance(sid, unicode):
                sid = sid.decode('utf8')
            keywords.add(cleanse(fid))
            keywords.add(cleanse(sid))

        fc[u'keywords'] = StringCounter(keywords)
        store.put([(cid, fc)])
        response.status = 201

        #return routes.v1_fc_put(request, response, lambda x: x, store, cid)


def new_folders(kvlclient, request):
    conf = {}
    if 'annotator_id' in request.query:
        conf['owner'] = request.query['annotator_id']
    return Folders(kvlclient, **conf)


feature_pretty_names = [
    ('ORGANIZATION', 'Organizations'), 
    ('PERSON', 'Persons'), 
    ('FACILITY', 'Facilities'),
    ('GPE', 'Geo-political Entities'),
    ('LOCATION', 'Locations'),
    ('SKYPE', 'Skype Handles'), 
    ('PHONE', 'Phone Numbers'), 
    ('email', 'Email Addresses'), 
    ('bowNP_unnorm', 'Noun Phrases'),
    ]

@app.post('/dossier/v1/highlighter/<cid>', json=True)
def v0_highlighter_post(request, response, tfidf, cid):
    '''Obtain highlights for a document POSTed as the body, which is the
    pre-design-thinking structure of the highlights API.  See v1 below.  

    NB: This end point will soon be deleted.

    The route for this endpoint is:
    ``POST /dossier/v0/highlighter/<cid>``.

    ``content_id`` is the id to associate with the given feature
    collection. The feature collection should be in the request
    body serialized as JSON.

    '''
    logger.info('got %r', cid)
    tfidf = tfidf or None
    content_type = request.headers.get('content-type', '')
    if not content_type.startswith('text/html'):
        logger.critical('content-type=%r', content_type)
        response.status = 415
        return {'error': {'code': 0, 'message': 'content_type=%r and should be text/html' % content_type}}

    url = urllib.unquote(cid.split('|', 1)[1])
    body = request.body.read()
    if len(body) == 0:
        response.status = 420
        return {'error': {'code': 1, 'message': 'empty body'}}
    logger.info('parsing %d bytes for url: %r', len(body), url)
    fc = etl.create_fc_from_html(url, body, tfidf=tfidf)
    if fc is None:
        logger.critical('failed to get FC using %d bytes from %r', len(body), url)
        response.status = 506
        return {'error': {'code': 2, 'message': 'FC not generated for that content'}}
    highlights = dict()
    for feature_name, pretty_name in feature_pretty_names:
        # Each type of string is 
        if feature_name not in fc: continue
        total = sum(fc[feature_name].values())
        highlights[pretty_name] = [
            (phrase, count / total, [], [])
            for phrase, count in sorted(fc[feature_name].items(), key=itemgetter(1), reverse=True)]
        logger.info('%r and %d keys', feature_name, len(highlights[pretty_name]))
    return {'highlights': highlights}


@app.post('/dossier/v1/highlights', json=True)
def v1_highlights_post(request, response, tfidf):
    '''Obtain highlights for a document POSTed inside a JSON object.  

    The route for this endpoint is:
    ``POST /dossier/v1/highlights``.

    The expected input structure is a JSON UTF-8 encoded string of an
    object with these keys:

    .. code-block:: javascript
      {
        // only text/html is supported at this time; hopefully PDF.js
        // enables this to support PDF rendering too.
        "content-type": "text/html",

        // URL of the page (after resolving all redirects)
        "content-location": "http://...",

        // If provided by the original host, this will be populated,
        // otherwise it is empty.
        "last-modified": "datetime string or empty string",

        // full page contents obtained by Javascript in the browser
        // extension accessing `document.documentElement.innerHTML` which
        // will already have been converted from `document.characterSet`
        // to unicode scalar values, and therefore can be serialized to
        // UTF-8 without issues.  (Needs testing on many browsers.)
        "body": "... the body content ...",
      }


    The output structure is a JSON UTF-8 encoded string of an
    object with these keys:

    .. code-block:: javascript
      {
        "highlights": [Highlight, Highlight, ...]
      }


    where a `Highlight` object has this structure:

    .. code-block:: javascript
      {
        // float in the range [0, 1]
        "score": 0.7

        // a string presented with a check box inside the options
        // bubble when the user clicks the extension icon to choose
        // which categories of highlights should be displayed.
        "category": "Organization",

        // `queries` are strings that are to be presented as
        // suggestions to the user, and the extension enables the user
        // to click any of the configured search engines to see
        // results for a selected query string.
        "queries": [],

        // zero or more strings to match in the document and highlight
        // with a single color.
        "strings": [],

        // zero or more XpathRange objects to lookup in the document
        // and highlight with a single color.
        "ranges": [],

        // zero or more regular expression strings to compile and 
        // execute to find spans to highlight with a single color.
        "regexes": []
      }

    where an XpathRange object is:

    .. code-block:: javascript
      {
        "start": XPathOffset,
        "end": XPathOffset
      }

    where an XpathOffset object is:

    .. code-block:: javascript
      {
        "node": "/html[1]/body[1]/p[1]/text()[2]",
        "idx": 4,
      }

    All of the `strings`, `ranges`, and `regexes` in a `Highlight`
    object should be given the same highlight color.  An `Highlight`
    object can provide values in any of the three `strings`, `ranges`,
    or `regexes` lists, and all should be highlighted.

    '''
    tfidf = tfidf or None
    content_type = request.headers.get('content-type', '')
    if not content_type.startswith('application/json'):
        logger.critical('content-type=%r', content_type)
        response.status = 415
        return {'error': {'code': 0, 
        'message': 'content_type=%r and should be text/html' % content_type}}

    body = request.body.read()
    if len(body) == 0:
        response.status = 400
        return {'error': {'code': 1, 'message': 'empty body'}}
    try:
        data = json.loads(body)
    except Exception, exc:
        response.status = 400
        return {'error': {'code': 2, 'message': 'failed to read JSON body: %s' % exc}}


    if not isinstance(data, dict):
        response.status = 400
        return {'error': {'code': 3, 'message': 'JSON payload deserialized to other than an object: %r' % type(data)}}

    expected_keys = set(['content-type', 'content-location', 'last-modified', 'body'])
    if set(data.keys()) != expected_keys:
        response.status = 400
        return {'error': {'code': 4, 'message': 'other than expected keys in JSON object.  Expected %r and received %r' % (expected_keys, set(data.keys()))}}

    if len(data['content-location']) < 3:
        response.status = 400
        return {'error': {'code': 5, 'message': 'received invalid content-location=%r' % data['content-location']}}

    if len(data['body']) < 3:
        response.status = 400
        return {'error': {'code': 6, 'message': 'received too little body=%r' % data['body']}}

    fc = etl.create_fc_from_html(data['content-location'], data['body'], tfidf=tfidf, encoding=None)
    if fc is None:
        logger.critical('failed to get FC using %d bytes from %r', len(body), url)
        response.status = 500
        return {'error': {'code': 2, 'message': 'FC not generated for that content'}}
    highlights = dict()
    for feature_name, pretty_name in feature_pretty_names:
        # Each type of string is 
        if feature_name not in fc: continue
        total = sum(fc[feature_name].values())
        highlights[pretty_name] = [
            (phrase, count / total)
            for phrase, count in sorted(fc[feature_name].items(), key=itemgetter(1), reverse=True)]
        logger.info('%r and %d keys', feature_name, len(highlights[pretty_name]))

    return {'highlights': build_highlight_objects(data['body'], highlights)}

def build_highlight_objects(html, highlights, uniformize_html=True):
    '''converts a dict of pretty_name --> [tuple(string, score), ...] to
    `Highlight` objects as specified above.

    '''
    if uniformize_html:
        try:
            html = uniform_html(html)
        except Exception, exc:
            logger.info('failed to get uniform_html(%d bytes) --> %s',
                        len(html), exc, exc_info=True)
            html = None

    highlight_objects = []
    for category, phrase_scores in highlights.iteritems():
        for (phrase, score) in phrase_scores:
            hl = dict(
                score=score,
                category=category,
                )
            ranges = make_xpath_ranges(html, phrase)
            if ranges:
                hl['ranges'] = ranges
            elif phrase in html:
                hl['strings'] = [phrase]
            else:
                hl['regexes'] = ['/%s/i' % phrase]
            highlight_objects.append(hl)
    return highlight_objects


def make_xpath_ranges(html, phrase):
    '''Given a HTML string and a `phrase`, build a regex to find offsets
    for the phrase, and then build a list of `XPathRange` objects for
    it.  If this fails, return empty list.

    '''
    if not html: return []
    if not isinstance(phrase, unicode):
        try:
            phrase = phrase.decode('utf8')
        except:
            logger.info('failed %r.decode("utf8")', exc_info=True)
            return []

    phrase_re = re.compile(phrase, flags=re.UNICODE | re.IGNORECASE | re.MULTILINE)
    spans = []
    for match in phrase_re.finditer(html, overlapped=False):
        spans.append(match.span()) # a list of tuple(start, end) char indexes

    # now run fancy aligner magic to get xpath info and format them as
    # XPathRange per above
    try:
        xpath_ranges = list(char_offsets_to_xpaths(html, spans))
    except:
        logger.info('failed to get xpaths', exc_info=True)
        return []
    ranges = []
    for xpath_range in filter(None, xpath_ranges):
        ranges.append(dict(
            start=dict(node=xpath_range.start_xpath,
                       idx=xpath_range.start_offset),
            end=dict(node=xpath_range.end_xpath,
                     idx=xpath_range.end_offset)))

    return ranges

