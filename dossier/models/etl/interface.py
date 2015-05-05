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

import abc
import time
import urllib

from streamcorpus_pipeline._clean_visible import cleanse, make_clean_visible
from streamcorpus_pipeline._clean_html import make_clean_html

from dossier.fc import FeatureCollection, StringCounter
import dossier.models.features as features


class ETL(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def cids_and_fcs(self):
        raise NotImplementedError


def mk_content_id(key):
    return 'web|' + urllib.quote(key, safe='~')


def html_to_fc(html, url=None, timestamp=None, other_features=None):
    def add_feature(name, xs):
        if name not in fc:
            fc[name] = StringCounter()
        fc[name] += StringCounter(xs)

    html = uni(html)
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
    fc[u'meta_url'] = uni(url)

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

    for feat_name, feat_val in other_features.iteritems():
        fc[feat_name] = feat_val

    return fc


def add_sip_to_fc(fc, tfidf, limit=40):
    if 'bowNP' not in fc:
        return
    sips = features.sip_noun_phrases(tfidf, fc['bowNP'].keys(), limit=limit)
    fc[u'bowNP_sip'] = StringCounter(sips)


def uni(s):
    if isinstance(s, str):
        try:
            return unicode(s, 'utf-8')
        except UnicodeDecodeError:
            return unicode(s, 'latin-1')
    return s
