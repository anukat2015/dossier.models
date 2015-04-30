'''
.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2014 Diffeo, Inc.

Generate feature collections with your data
===========================================
This library ships with a command line program ``dossier.etl`` which
provides a rudimentary pipeline for transforming data from your database
to feature collections managed by :mod:`dossier.store`.

(Currently, ``dossier.etl`` is hard-coded to support a specific HBase
database, but it will be generalized as part of future work.)
'''
from __future__ import absolute_import, division, print_function

import argparse
from collections import defaultdict
from itertools import chain, islice
import multiprocessing
import re
import sys
import time
import traceback
import urllib
import zlib

import cbor
from gensim import corpora, models
import happybase

from dossier.fc import FeatureCollection, FeatureCollectionChunk, StringCounter
from dossier.store import Store
from dossier.web import streaming_sample
import kvlayer
from streamcorpus_pipeline._clean_visible import cleanse, make_clean_visible
from streamcorpus_pipeline._clean_html import make_clean_html
import yakonfig

import dossier.models.features as features


def html_to_fc(html, url=None, timestamp=None, other_features=None):
    def add_feature(name, xs):
        if name not in fc:
            fc[name] = StringCounter()
        fc[name] += StringCounter(xs)

    if isinstance(html, str):
        html = unicode(html, 'utf-8')
    timestamp = timestamp or int(time.time() * 1000)
    other_features = other_features or {}
    url = url or ''

    clean_html = make_clean_html(html.encode('utf-8')).decode('utf-8')
    clean_vis = make_clean_visible(clean_html.encode('utf-8')).decode('utf-8')

    fc = FeatureCollection()
    fc[u'meta_raw'] = html
    fc[u'meta_clean_html'] = clean_html
    fc[u'meta_clean_visible'] = clean_vis
    fc[u'meta_timestamp'] = unicode(timestamp)
    fc[u'meta_url'] = unicode(url, 'utf-8')
    for feat_name, feat_val in other_features.iteritems():
        fc[feat_name] = feat_val

    add_feature(u'phone', features.phones(clean_vis))
    add_feature(u'email', features.emails(clean_vis))
    add_feature(u'bowNP', features.noun_phrases(cleanse(clean_vis)))

    add_feature(u'image_url', features.image_urls(clean_html))
    add_feature(u'a_url', features.a_urls(clean_html))

    ## get parsed versions, extract usernames
    fc[u'img_url_path_dirs'] = features.path_dirs(fc[u'image_url'])
    fc[u'img_url_hostnames'] = features.host_names(fc[u'image_url'])
    fc[u'img_url_usernames'] = features.usernames(fc[u'image_url'])
    fc[u'a_url_path_dirs'] = features.path_dirs(fc[u'a_url'])
    fc[u'a_url_hostnames'] = features.host_names(fc[u'a_url'])
    fc[u'a_url_usernames'] = features.usernames(fc[u'a_url'])

    return fc


def add_sip_to_fc(fc, tfidf, limit=40):
    if 'bowNP' not in fc:
        return
    sips = features.sip_noun_phrases(tfidf, fc['bowNP'].keys(), limit=limit)
    fc[u'bowNP_sip'] = StringCounter(sips)


def row_to_content_obj(key_row):
    '''Returns ``FeatureCollection`` given an HBase artifact row.

    Note that the FC returned has a Unicode feature ``artifact_id``
    set to the row's key.
    '''
    key, row = key_row
    cid = 'web|' + urllib.quote(key, safe='~').encode('utf-8')
    response = row.get('response', {})

    other_bows = defaultdict(StringCounter)
    for attr, val in row.get('indices', []):
        other_bows[attr][val] += 1
    try:
        artifact_id = key
        if isinstance(artifact_id, str):
            artifact_id = unicode(artifact_id, 'utf-8')
        fc = html_to_fc(
            response.get('body', ''),
            url=row.get('url'), timestamp=row.get('timestamp'),
            other_features=dict(other_bows, **{'artifact_id': artifact_id}))
    except:
        fc = None
        print('Could not create FC for %s:' % cid, file=sys.stderr)
        print(traceback.format_exc())
    return cid, fc


def get_artifact_rows(conn, limit=5, start_key=None, stop_key=None):
    t = conn.table('artifact')
    scanner = t.scan(row_start=start_key, row_stop=stop_key,
                     limit=limit, batch_size=20)
    for key, data in scanner:
        yield key, unpack_artifact_row(data)


def unpack_artifact_row(row):
    data = {
        'url': row['f:url'],
        'timestamp': int(row['f:timestamp']),
        'request': {
            'method': row['f:request.method'],
            'client': cbor.loads(zlib.decompress(row['f:request.client'])),
            'headers': cbor.loads(zlib.decompress(row['f:request.headers'])),
            'body': cbor.loads(zlib.decompress(row['f:request.body'])),
        },
        'response': {
            'status': row['f:response.status'],
            'server': {
                'hostname': row['f:response.server.hostname'],
                'address': row['f:response.server.address'],
            },
            'headers': cbor.loads(zlib.decompress(row['f:response.headers'])),
            'body': cbor.loads(zlib.decompress(row['f:response.body'])),
        },
        'indices': [],
    }
    for kk, vv in row.items():
        mm = re.match(r"^f:index\.(?P<key>.*)\.[0-9]+$", kk)
        if mm is not None:
            data['indices'].append((mm.group('key'), vv))
    return data


def unpack_noun_phrases(row):
    body = cbor.loads(zlib.decompress(row['f:response.body']))
    body = make_clean_visible(body.encode('utf-8')).decode('utf-8')
    body = cleanse(body)
    return features.noun_phrases(body)


def generate_fcs(tfidf, get_conn, pool, add, limit=5, batch_size=100,
                 start_key=None, stop_key=None):
    rows = get_artifact_rows(get_conn, limit=limit,
                             start_key=start_key, stop_key=stop_key)
    batch = []
    for i, (cid, fc) in enumerate(pool.imap(row_to_content_obj, rows), 1):
        if fc is None:
            continue
        add_sip_to_fc(fc, tfidf)
        if not any(cid == cid2 for cid2, _ in batch):
            # Since we can restart the scanner, we may end up regenerating
            # FCs for the same key in the same batch. This results in
            # undefined behavior in kvlayer.
            batch.append((cid, fc))

        if len(batch) >= batch_size:
            add(batch)
            batch = []
        if i % 100 == 0:
            status('%d of %s done'
                   % (i, 'all' if limit is None else str(limit)))
    if len(batch) > 0:
        add(batch)


def status(*args, **kwargs):
    kwargs['end'] = ''
    args = list(args)
    args[0] = '\033[2K\r' + args[0]
    print(*args, **kwargs)
    sys.stdout.flush()


def batch_iter(n, iterable):
    iterable = iter(iterable)
    while True:
        yield chain([next(iterable)], islice(iterable, n-1))
