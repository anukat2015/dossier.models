'''web service endpoints for supporting SortingDesk

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.
'''
from __future__ import absolute_import, division, print_function

from collections import defaultdict
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import logging
import os.path as path
import urllib

import bottle
import cbor
import json
from streamcorpus_pipeline import cleanse

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


@app.post('/dossier/v1/highlighter/<cid>', json=True)
def v1_highlighter_post(request, response, tfidf, cid):
    '''Obtain highlights for a document POSTed as the body

    The route for this endpoint is:
    ``POST /dossier/v1/highlighter/<cid>``.

    ``content_id`` is the id to associate with the given feature
    collection. The feature collection should be in the request
    body serialized as JSON.

    '''
    tfidf = tfidf or None
    content_type = request.headers.get('content-type', '')
    if not content_type.startswith('text/html'):
        logger.critical('content-type=%r', content_type)
        return

    url = urllib.unquote(cid.split('|', 1)[1])
    logger.info('parsing url: %r', url)
    fc = etl.create_fc_from_html(url, request.body.read(), tfidf=tfidf)
    keys = set()
    for fname in ['ORGANIZATION', 'PERSON', 'FACILITY', 'GPE', 'LOCATION',
                  'SKYPE', 'PHONE', 'email', 'bowNP_unnorm']:
        keys.update(fc.get(fname, {}).keys())
        logger.info('%r and %d keys', fname, len(keys))
    return {'highlights': list(keys)}
