from __future__ import absolute_import, division, print_function

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import logging
import os.path as path
import urllib

from bs4 import BeautifulSoup
import bottle

from dossier.fc import StringCounter
from dossier.models import etl
from dossier.models.folder import Folders
from dossier.models.report import ReportGenerator
import dossier.web.routes as routes


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
    title = soup_get(soup, 'title', lambda v: v.get_text())
    body = soup_get(soup, 'body', lambda v: v.prettify())
    fc = etl.html_to_fc(body, url=url, other_features={
        u'title': title,
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
