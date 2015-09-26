'''Tests for the highlights endpoint

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.
'''

from cStringIO import StringIO
import json
from dossier.models.tests import kvl
from dossier.models.web.routes import make_xpath_ranges, \
    build_highlight_objects, \
    v1_highlights_post


good_html = '''
<html>
  <body>
    <h1>Cats</h1>
    <p>    </p>
    <ul>
      <li>Fluffy</li>
      <li>Fluffier</li>
    </ul>
  </body>
</html>
'''

def test_make_xpath_ranges():
    ranges = make_xpath_ranges(good_html, 'fluffy')

    assert len(ranges) == 1
    assert ranges[0]['start']['node'] == u'/html[1]/body[1]/ul[1]/li[1]/text()[1]'
    assert ranges[0]['start']['idx'] == 0
    assert ranges[0]['end']['node'] == u'/html[1]/body[1]/ul[1]/li[1]/text()[1]'
    assert ranges[0]['end']['idx'] == 6


    ranges = make_xpath_ranges(good_html, 'fluff')

    assert len(ranges) == 2
    assert ranges[0]['start']['node'] == u'/html[1]/body[1]/ul[1]/li[1]/text()[1]'
    assert ranges[0]['start']['idx'] == 0
    assert ranges[0]['end']['node'] == u'/html[1]/body[1]/ul[1]/li[1]/text()[1]'
    assert ranges[0]['end']['idx'] == 5

    assert ranges[1]['start']['node'] == u'/html[1]/body[1]/ul[1]/li[2]/text()[1]'
    assert ranges[1]['start']['idx'] == 0
    assert ranges[1]['end']['node'] == u'/html[1]/body[1]/ul[1]/li[2]/text()[1]'
    assert ranges[1]['end']['idx'] == 5



bad_html = '''
<html>
  <body>
    <h1>Cats</h1>
    <p>
    <ul>
      <li>Fluffy
      <li>Fluffier
    </ul></p>
  </body>
</html>
'''

def test_build_highlight_objects():
    highlights = build_highlight_objects(bad_html, {'cats': [('fluff', .9)]})
    assert len(highlights) == 1
    assert len(highlights[0]['xranges']) == 2


def test_build_highlight_objects_without_uniform():
    highlights = build_highlight_objects(bad_html, {'cats': [('fluff', .9)]},
                                         uniformize_html=False)
    assert len(highlights) == 1
    assert len(highlights[0]['regexes']) == 1


class Empty(object):
    pass

def test_v1_highlights_post(kvl):
    request = Empty()
    request.headers = {'content-type': 'application/json'}
    data = {
        'no-cache': True,
        'body': bad_html,
        'content-location': 'fooooo',
        'content-type': 'text/html',
        'last-modified': '',
        }
    request.body = StringIO(json.dumps(data))
    response = None
    tfidf = None
    results = v1_highlights_post(request, response, kvl, tfidf)

    assert results
    assert len(results['highlights']) == 2
    assert len(results['highlights'][0]['regexes']) == 1
    assert len(results['highlights'][1]['xranges']) == 1
