'''
Web services for active learning search engines
===============================================
This library ships with a command line application, ``dossier.models``,
that runs a web service. It is easy to run::

    dossier.models -c config.yaml

This runs it on ``localhost:8080``. A sample config file looks like:

.. code-block:: yaml

    dossier.store:
      feature_indexes: ['bowNP_sip', 'phone', 'region']

    dossier.models:
      tfidf_path: /path/to/tfidf/model.tfidf

    kvlayer:
      app_name: dossierstack
      namespace: myapp
      storage_type: redis
      storage_addresses: ['localhost:6379']

Where the ``tfidf_path`` corresponds to a TF-IDF model generated by
``gensim``. You can create one with the ``dossier.etl tfidf`` command.
(Note that this is optional!)

Once ``dossier.models`` is running, you can try the
Sorting Desk browser extension for Google Chrome:
https://chrome.google.com/webstore/detail/sorting-desk/ikcaehokdafneaiojndpmfbimilmlnid.
Once the extension is installed, you'll need to go to
its options and configure it to point to ``http://localhost:8080``.

Alternatively, you can follow the steps here to get a simple example
working on a sample data set:
https://github.com/dossier/dossier.models#running-a-simple-example.


Web service endpoints
---------------------
The web service has all of the same endpoints as :mod:`dossier.web`.
A couple of the endpoints are slightly enhanced to take advantage of
``dossier.models`` pairwise learning algorithm and feature extraction.
These enhanced endpoints are :func:`dossier.web.routes.v1_search` and
:func:`dossier.web.routes.v1_fc_put`. The ``v1_search`` endpoint is
enhanced with the addition of the :func:`dossier.models.similar` and
:func:`dossier.models.dissimilar` search engines. Additionally, the
``v1_fc_put`` accepts ``text/html`` content and will generate a feature
collection for you.


How the Sorting Desk browser extension works
--------------------------------------------
SortingDesk needs a ``dossier.models`` web server in order to function
properly. Namely, it uses ``dossier.models`` (and the underlying
DossierStack) to add/update feature collections, store ground truth
data as "labels," and run pairwise learning algorithms to rank relevant
search results. All of this information is saved to the underlying
database, so it is persistent across all user sessions.

``dossier.models`` also provides a folder/sub-folder organization
UI. Currently, this is built on top of Google Chrome's local storage.
Namely, it doesn't yet use the folder/subfolder web services described
in :mod:`dossier.web`. Therefore, folders/sub-folders don't yet persist
across multiple user sessions, but they will once we migrate off of
Chrome's local storage.
'''
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
            return routes.v1_fc_put(request, response, lambda x: x, store, cid)


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


def same_subfolder(store, label_store):
    '''Filter out results in the same subfolder.'''
    folders = web.Folders(store, label_store)
    def init_filter(query_content_id):
        subfolders = folders.parent_subfolders(query_content_id)
        cids = set()
        for folder_id, subfolder_id in subfolders:
            for cid, subid in folders.items(folder_id, subfolder_id):
                cids.add(cid)

                # Also add directly connected labels too.
                for lab in label_store.directly_connected((cid, subid)):
                    cids.add(lab.other(cid))

        def p((content_id, fc)):
            return content_id not in cids
        return p
    return init_filter


def get_application():
    engines = {
        'dissimilar': dissimilar,
        'similar': similar,
        'random': web.engine_random,
        'index_scan': web.engine_index_scan,
    }
    args, application = web.get_application(
        routes=[add_routes], search_engines=engines,
        filter_preds={'already_labeled': same_subfolder})

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
