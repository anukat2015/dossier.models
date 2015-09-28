'''Tests for the highlights endpoint

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.
'''

from cStringIO import StringIO
import json
import logging
import pytest
import time

from dossier.models.tests import kvl
from dossier.models.tests.test_web import tfidf, tfidf_path
from dossier.models.web.routes import make_xpath_ranges, \
    highlights_worker, \
    build_highlight_objects, \
    v1_highlights_post, \
    v1_highlights_get

logger = logging.getLogger(__name__)

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


class MockTaskMaster(object):
    def __init__(self, kvl, tfidf_path):
        self.config = {
            'kvlayer': kvl._config,
            'dossier.models': {'tfidf_path': tfidf_path},
            }

    def add_work_units(self, name_space, work_units):
        '''This mock of :class:`coordinate.TaskMaster` calls
        `highlights_worker` directly instead of letting the
        `coordinate_worker` run it asynchronously.  To do this, it
        builds a sufficiently real duck-typed
        :class:`coordinate.WorkUnit` that it can run
        `highlights_worker` in its normal way.

        '''
        wu = Empty()
        wu.spec = {'config': self.config}
        wu.key = work_units[0][0]
        highlights_worker(wu)


class Empty(object):
    pass

def basic_post(kvl, tfidf, allow_store, mock_tm):
    request = Empty()
    request.headers = {'content-type': 'application/json'}
    data = {
        'store': allow_store,
        'body': bad_html,
        'content-location': 'fooooo',
        'content-type': 'text/html',
        'last-modified': '',
        }
    request.body = StringIO(json.dumps(data))
    response = Empty()
    results = v1_highlights_post(request, response, kvl, tfidf, 
                                 min_delay=0, task_master=mock_tm)

    max_loops = 10
    loops = 0
    while results['state'] == 'pending' and loops < max_loops:
        loops += 1
        results = v1_highlights_get(response, kvl, results['id'])
        logger.info(results)
        time.sleep(0.1)

    assert results
    assert len(results['highlights']) == 2
    assert len(results['highlights'][0]['regexes']) == 1
    assert len(results['highlights'][1]['xranges']) == 1

    return results

steps = [
    (False, 'completed'),
    (True, 'stored'),
    (True, 'stored'),
    (True, 'stored'),
    (False, 'completed'),
    (False, 'completed'),
    (True, 'stored'),
    (True, 'stored'),
    (True, 'stored'),
    (False, 'completed'),
]

def test_v1_highlights_post(kvl, tfidf_path, tfidf):
    mock_tm = MockTaskMaster(kvl, tfidf_path)
    for idx, (allow_store, state) in enumerate(steps):
        logger.info('step=%d, allow_store=%r, state=%r', idx, allow_store, state)
        results = basic_post(kvl, tfidf, allow_store, mock_tm)
        assert results['state'] == state
