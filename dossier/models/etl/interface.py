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
from itertools import chain
import json
import logging
import time
import urllib

from bs4 import BeautifulSoup
import gensim
import kvlayer
from dossier.store import Store
from streamcorpus import Chunk
from streamcorpus_pipeline.stages import Configured
from streamcorpus_pipeline._clean_visible import cleanse, make_clean_visible
from streamcorpus_pipeline._clean_html import make_clean_html
import yakonfig

from dossier.fc import FeatureCollection, StringCounter
import dossier.models.features as features

logger = logging.getLogger(__name__)

class ETL(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def cids_and_fcs(self):
        raise NotImplementedError


def create_fc_from_html(url, html, encoding='utf-8', tfidf=None, other_features=None):
    if encoding is not None:
        html = unicode(html, encoding)
    soup = BeautifulSoup(html, "lxml")
    title = soup_get(soup, 'title', lambda v: v.get_text())
    body = soup_get(soup, 'body', lambda v: v.prettify())
    if other_features is None:
        other_features = {}
    other_features.update({
        u'title': StringCounter([title]),
        u'titleBow': StringCounter(title.split()),
    })
    fc = html_to_fc(body, url=url, other_features=other_features)
    if fc is None:
        return None
    if tfidf is not None:
        add_sip_to_fc(fc, tfidf)
    return fc


def soup_get(soup, sel, cont):
    v = soup.find(sel)
    if v is None:
        return u''
    else:
        return cont(v)


class to_dossier_store(Configured):
    '''A :mod:`streamcorpus_pipeline` `writer` stage with one optional parameter:

    .. code-block:: yaml

        tfidf_path: path/to/tfidf.data

    '''
    config_name = 'to_dossier_store'
    default_config = {
        'tfidf_path': None
    }
    def __init__(self, *args, **kwargs):
        super(to_dossier_store, self).__init__(*args, **kwargs)
        kvl = kvlayer.client()
        feature_indexes = None
        try:
            conf = yakonfig.get_global_config('dossier.store')
            feature_indexes = conf['feature_indexes']
        except KeyError:
            pass
        self.store = Store(kvl,
                           feature_indexes=feature_indexes)
        tfidf_path = self.config.get('tfidf_path')
        self.tfidf = gensim.models.TfidfModel.load(tfidf_path)

    def process(self, t_path, name_info, i_str):
        '''converts each :attr:`streamcorpus.StreamItem.body.clean_html` from
        `t_path` into a :class:`~dossier.fc.FeatureCollection` and saves it in
        a :class:`~dossier.store.Store` configured with the global `kvlayer`
        config.

        '''
        def cids_and_fcs():
            count = 0
            seen = set()
            for si in Chunk(t_path):
                clean_html = getattr(si.body, 'clean_html', '')
                if clean_html is None or len(clean_html.strip()) == 0:
                    logger.warn('dropping SI lacking clean_html: %r', si.abs_url)
                    continue
                if 'other_features' in si.other_content:
                    other_features = json.loads(si.other_content['other_features'].raw)
                else:
                    other_features = None
                fc = html_to_fc(
                    clean_html=si.body.clean_html.decode('utf-8'),
                    clean_visible=si.body.clean_visible.decode('utf-8'),
                    encoding='utf-8',
                    url=si.abs_url,
                    timestamp=si.stream_time.epoch_ticks,
                    other_features=other_features,
                )
                add_sip_to_fc(fc, self.tfidf)
                content_id = mk_content_id(str(fc.get(u'meta_url')))
                if content_id in seen:
                    logger.warn('dropping duplicate content_id=%r', content_id)
                else:
                    seen.add(content_id)
                    yield content_id, fc
                    count += 1
            logger.info('saved %d FCs from %d SIs', count, len(seen))
        self.store.put(cids_and_fcs())
        ## interface spec of streamcorpus_pipeline writers requires
        ## returning a list of locally generated paths.
        return []

    __call__ = process


def mk_content_id(key):
    return 'web|' + urllib.quote(key, safe='~')


def html_to_fc(html=None, clean_html=None, clean_visible=None, encoding=None, url=None,
               timestamp=None, other_features=None):
    '''`html` is expected to be a raw string received over the wire from a
    remote webserver, and `encoding`, if provided, is used to decode
    it.  Typically, encoding comes from the Content-Type header field.
    The :func:`~streamcorpus_pipeline._clean_html.make_clean_html`
    function handles character encodings.

    '''
    def add_feature(name, xs):
        if name not in fc:
            fc[name] = StringCounter()
        fc[name] += StringCounter(xs)

    timestamp = timestamp or int(time.time() * 1000)
    other_features = other_features or {}

    if clean_html is None:
        if html is not None:
            try:
                clean_html_utf8 = make_clean_html(html, encoding=encoding)
            except: 
                logger.warn('dropping doc because:', exc_info=True)
                return
            clean_html = clean_html_utf8.decode('utf-8')
        else:
            clean_html_utf8 = u''
            clean_html = u''
    else:
        clean_html_utf8 = u''

    if clean_visible is None or len(clean_visible) == 0:
        clean_visible = make_clean_visible(clean_html_utf8).decode('utf-8')
    elif isinstance(clean_visible, str):
        clean_visible = clean_visible.decode('utf-8')

    fc = FeatureCollection()
    fc[u'meta_raw'] = html and uni(html, encoding) or u''
    fc[u'meta_clean_html'] = clean_html
    fc[u'meta_clean_visible'] = clean_visible
    fc[u'meta_timestamp'] = unicode(timestamp)

    url = url or u''

    fc[u'meta_url'] = uni(url)

    add_feature(u'icq', features.ICQs(clean_visible))
    add_feature(u'skype', features.skypes(clean_visible))
    add_feature(u'phone', features.phones(clean_visible))
    add_feature(u'email', features.emails(clean_visible))
    bowNP, normalizations = features.noun_phrases(
        cleanse(clean_visible), included_unnormalized=True)
    add_feature(u'bowNP', bowNP)
    bowNP_unnorm = chain(*normalizations.values())
    add_feature(u'bowNP_unnorm', bowNP_unnorm)

    add_feature(u'image_url', features.image_urls(clean_html))
    add_feature(u'a_url', features.a_urls(clean_html))

    ## get parsed versions, extract usernames
    fc[u'img_url_path_dirs'] = features.path_dirs(fc[u'image_url'])
    fc[u'img_url_hostnames'] = features.host_names(fc[u'image_url'])
    fc[u'usernames'] = features.usernames(fc[u'image_url'])

    fc[u'a_url_path_dirs'] = features.path_dirs(fc[u'a_url'])
    fc[u'a_url_hostnames'] = features.host_names(fc[u'a_url'])

    fc[u'usernames'] += features.usernames(fc[u'a_url'])

    #fc[u'usernames'] += features.usernames2(
    #    fc[u'meta_clean_visible'])

    # beginning of treating this as a pipeline...
    xform = features.entity_names()
    fc = xform.process(fc)

    for feat_name, feat_val in other_features.iteritems():
        fc[feat_name] += StringCounter(feat_val)

    return fc


def add_sip_to_fc(fc, tfidf, limit=40):
    '''add "bowNP_sip" to `fc` using `tfidf` data
    '''
    if 'bowNP' not in fc:
        return
    if tfidf is None:
        return
    sips = features.sip_noun_phrases(tfidf, fc['bowNP'].keys(), limit=limit)
    fc[u'bowNP_sip'] = StringCounter(sips)


def uni(s, encoding=None):
    # unicode string feat
    if not isinstance(s, unicode):
        try:
            return unicode(s, encoding)
        except:
            try:
                return unicode(s, 'utf-8')
            except UnicodeDecodeError:
                return unicode(s, 'latin-1')
    return s
