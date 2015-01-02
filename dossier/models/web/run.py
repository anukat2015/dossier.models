from __future__ import absolute_import, division, print_function

import json
import logging
import os.path as path
import urllib

from bs4 import BeautifulSoup
import bottle
from gensim import models
import requests

from dossier.fc import FeatureCollection, StringCounter
from dossier.models import etl
from dossier.models.pairwise import dissimilar, similar
import dossier.web as web


logger = logging.getLogger(__name__)
web_static_path = path.join(path.split(__file__)[0], 'static')
bottle.TEMPLATE_PATH.insert(0, path.join(web_static_path, 'tpl'))


def add_routes(app):
    @app.get('/SortingQueue')
    def example_sortingqueue():
        return bottle.template('example-sortingqueue.html')

    @app.get('/SortingDesk')
    def example_sortingdesk():
        return bottle.template('example-sortingdesk.html')

    @app.get('/static/<name:path>')
    def v1_static(name):
        return bottle.static_file(name, root=web_static_path)

    @app.put('/dossier/v1/feature-collection/<cid>')
    def v1_fc_put(request, response, store, cid):
        '''Store a single feature collection.

        The route for this endpoint is:
        ``PUT /dossier/v1/feature-collections/<content_id>``.

        ``content_id`` is the id to associate with the given feature
        collection. The feature collection should be in the request
        body serialized as JSON.

        This endpoint returns status ``201`` upon successful storage.
        An existing feature collection with id ``content_id`` is
        overwritten.
        '''
        if request.headers.get('content-type', '').startswith('text/html'):
            url = urllib.unquote(cid.split('|', 1)[1])
            fc = create_fc_from_html(url, request.body.getvalue())
        else:
            fc = FeatureCollection.from_dict(json.load(request.body))
        store.put([(cid, fc)])
        response.status = 201


class index_scan_make_fc(web.engine_index_scan):
    '''Index scan with FC materialization.

    This search engine knows the format of a particular breed of content
    id, which enables this search engine to create FCs for that id.

    Namely, if a content_id starts with ``web|``, then everything following
    that prefix is a URL. We try to do a very basic and minimal crawl of
    that URL to produce an FC to query with.
    '''
    def __init__(self, store, tfidf_path=None):
        self.store = store
        self._tfidf_path = tfidf_path
        self._tfidf = None

    @property
    def tfidf(self):
        # This lazily loads the TF-IDF model. This is done *every* time a new
        # FC is made. This is probably OK because FCs are not made this way
        # in bulk, only when the user specifically requests to search with
        # a query that doesn't have an FC.
        #
        # On the other hand, if the model is sufficiently large, that could
        # negatively affect latency. I'm calling that a premature optimization
        # for now. ---AG
        if self._tfidf is None and self._tfidf_path is not None:
            self._tfidf = models.TfidfModel.load(self._tfidf_path)
        return self._tfidf

    def get_query_fc(self, content_id):
        fc = super(index_scan_make_fc, self).get_query_fc(content_id)
        if fc is not None:
            return fc
        if not content_id.startswith('web|'):
            return None
        url = urllib.unquote(content_id.split('|', 1)[1])
        html = requests.get(url).text
        fc = create_fc_from_html(url, html, tfidf=self.tfidf)
        self.store.put([(content_id, fc)])
        return fc


def create_fc_from_html(url, html, tfidf=None):
    soup = BeautifulSoup(html)
    title = soup.find('title').get_text().decode('utf-8')
    fc = etl.html_to_fc(html, url=url, other_features={
        u'title': title,
        u'titleBow': StringCounter(title.split()),
    })
    if fc is None:
        return None
    if tfidf is not None:
        etl.add_sip_to_fc(fc, tfidf)
    return fc


def get_application():
    engines = {
        'dissimilar': dissimilar,
        'similar': similar,
        'random': web.engine_random,
        'index_scan': index_scan_make_fc,
    }
    args, application = web.get_application(routes=[add_routes],
                                            search_engines=engines)
    return args, application


def main():
    args, application = get_application()
    web.run_with_argv(args, application)


if __name__ == '__main__':
    main()
