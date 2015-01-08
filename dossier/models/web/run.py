from __future__ import absolute_import, division, print_function

import logging
import os.path as path
import urllib

from bs4 import BeautifulSoup
import bottle
try:
    from gensim import models
    TFIDF = True
except ImportError:
    TFIDF = False

from dossier.fc import StringCounter
from dossier.models import etl
from dossier.models.pairwise import dissimilar, similar
import dossier.web as web
import dossier.web.config as config
import dossier.web.routes as routes
import yakonfig


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
            logger.info('created FC for "%r": %r', cid, fc)
            store.put([(cid, fc)])
            return routes.fc_to_json(fc)
        else:
            return routes.v1_fc_put(request, response, store, cid)


def create_fc_from_html(url, html, tfidf=None):
    soup = BeautifulSoup(unicode(html, 'utf-8'))
    title = soup.find('title').get_text()
    body = soup.find('body').prettify()
    fc = etl.html_to_fc(body, url=url, other_features={
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
        'index_scan': web.engine_index_scan,
    }
    args, application = web.get_application(routes=[add_routes],
                                            search_engines=engines)

    tfidf_model = False
    if TFIDF:
        try:
            conf = yakonfig.get_global_config('dossier.models')
            tfidf_path = conf['tfidf_path']
            tfidf_model = models.TfidfModel.load(tfidf_path)
        except KeyError:
            pass
    application.install(
        config.create_injector('tfidf', lambda: tfidf_model))
    return args, application


def main():
    args, application = get_application()
    web.run_with_argv(args, application)


if __name__ == '__main__':
    main()
