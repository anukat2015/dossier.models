'''``dossier.models.features.basic`` provides simple transforms that
construct features.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2015 Diffeo, Inc.

'''
import json
import pytest

from dossier.models.tests import nltk_data, example_fc
from dossier.fc import StringCounter, FeatureCollection
import dossier.models.features as features

def test_extract_phones():
    txt = '''
Phone: 111-222-3333
Phone: 1112223333
Phone: 1-111-222-3333
Phone: 11112223333
Phone: 222-3333
Phone: 2223333
'''
    assert StringCounter(features.phones(txt)) == StringCounter({
        '1112223333': 2,
        '11112223333': 2,
        '2223333': 2,
    })


def test_a_urls():
    html = '''
<a href="http://ExAmPle.com/My Page.html">
<a href="http://example.com/My%20Page.html">
'''
    assert StringCounter(features.a_urls(html)) == StringCounter({
        'http://example.com/My Page.html': 2,
    })

def test_image_urls():
    html = '''
<img src="http://ExAmPle.com/My Image.jpg">
<img src="http://example.com/My%20Image.jpg">
'''
    assert StringCounter(features.image_urls(html)) == StringCounter({
        'http://example.com/My Image.jpg': 2,
    })


def test_extract_emails():
    txt = '''
email: abc@example.com
email: AbC@eXamPle.com
'''
    assert StringCounter(features.emails(txt)) == StringCounter({
        'abc@example.com': 2,
    })

def test_host_names():
    urls = StringCounter()
    urls['http://www.example.com/folder1'] = 3
    urls['http://www.example.com/folder2'] = 2
    urls['http://www.different.com/folder2'] = 7


    assert features.host_names(urls) == StringCounter({
        'www.example.com': 5,
        'www.different.com': 7,
    })

def test_path_dirs():
    urls = StringCounter()
    urls['http://www.example.com/folder1/folder3/index.html?source=dummy'] = 3
    urls['http://www.example.com/folder2/folder1'] = 2
    urls['http://www.different.com/folder2'] = 7


    assert features.path_dirs(urls) == StringCounter({
        'folder1': 5,
        'folder2': 9,
        'folder3': 3,
        'index.html': 3,
    })

def test_entity_names(example_fc, nltk_data):
    '''test for the `entity_names` transform
    '''
    xform = features.entity_names()
    fc = xform.process(example_fc)
    assert u'PERSON' in fc, fc.keys()
    assert u'craig winton' in fc[u'PERSON']
    #print(json.dumps(dict(fc), indent=4))
