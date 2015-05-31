'''``dossier.models.features.basic`` provides simple transforms that
construct features.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2015 Diffeo, Inc.

'''
import json
import pytest

import nltk

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


example_usernames_from_paths = [
     (r'http://www.example.com/user/folder3/index.html?source=dummy', 'folder3', 3),
     (r'http://www.example.com/user/myaccount', 'myaccount', 2),
     (r'http://www.different.com/folder3', None, 4),
     (r'http://www.different.com/user/myaccount', 'myaccount', 7),
     (r'http://www.also.com/user', None, 23),
     (r'http://www.also2.com/user/user', 'user', 1),
     (r'http://frob.com/user/my_account/media/Dresses/hi.jpg', 'my_account', 1),
     (r'https://www.facebook.com/my_account', 'my_account', 1),
     (r'https://twitter.com/my_account', 'my_account', 1),
     (r'C:\WINNT\Profiles\myaccount%MyUserProfile%', 'myaccount', 3), # Microsoft Windows NT
     (r'C:\WINNT\Profiles\myaccount', 'myaccount', 3), # Microsoft Windows NT
     (r'd:\WINNT\Profiles\myaccount', 'myaccount', 3), # Microsoft Windows NT
     (r'X:\Documents and Settings\myaccount', 'myaccount', 8), # Microsoft Windows 2000, XP and 2003
     (r'C:\Users\myaccount', 'myaccount', 3), # Microsoft Windows Vista, 7 and 8
     (r'C:\Users\myaccount\dog', 'myaccount', 3), # Microsoft Windows Vista, 7 and 8
     (r'C:\Users\whg\Desktop\Plug\FastGui(LYT)\Shell\Release\Shell.pdb', 'whg', 2),
     (r'C:\Documents and Settings\whg\\Plug\FastGui(LYT)\Shell\Release\Shell.pdb', 'whg', 3),
     (r'C:\Users\whg\Desktop\Plug\FastGui(LYT)\Shell\Release\Shell.pdb', 'whg', 3),
     (r'/home/myaccount$HOME', 'myaccount', 5), # Unix-Based
     (r'/var/users/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/u01/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/user/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/users/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/var/users/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/home/myaccount', 'myaccount', 3), # Linux / BSD (FHS)
     (r'/Users/my_account$HOME', 'my_account', 5), # Mac OS X
     (r'/Users/my_account', 'my_account', 5), # Mac OS X
     (r'/data/media/myaccount', 'myaccount', 5) # Android
     ]

@pytest.mark.parametrize(
    ('url_or_path', 'username', 'count'),
    example_usernames_from_paths
)
def test_usernames(url_or_path, username, count):
    urls = StringCounter()
    urls[url_or_path] += count

    if username is not None:
        results = features.usernames(urls)
        assert results == StringCounter({username: count})


example_text = u'''Summer School of the Arts filling fast\nWanganui people have the chance to learn the intricacies of decorative sugar art from one of the country\xe2\x80\x99s top pastry chefs at Whanganui UCOL\xe2\x80\x99s Summer School of the Arts in January.\nTalented Chef de Partie, Adele Hingston will take time away from her duties at Christchurch\xe2\x80\x99s Crowne Plaza to demonstrate the tricks and techniques of cake decorating including creating flower sprays and an introduction to royal icing.\nDemand has been high for places in the Summer School of the Arts but there are still opportunities for budding artists to hone their skills in subjects as diverse as jewellery making, culinary sugar art and creative writing. \n\xe2\x80\x9cThe painting, pattern drafting and hot glass classes filled almost immediately,\xe2\x80\x9d says Summer School Coordinator Katrina Langdon. \xe2\x80\x9cHowever there are still places available in several of the programmes.\xe2\x80\x9d\nEighteen distinguished artists will each share their particular creative talents during week long programmes in painting, writing, drawing, jewellery, fibre arts, printmaking, photography, sculpture, glass, fashion and culinary arts.\n\xe2\x80\x9cI suggest anyone who is considering joining us for the Summer School should register now. January will be here before we know it,\xe2\x80\x9d says Katrina.\nWhanganui UCOL Summer School of the Arts runs from 10-16 January 2010. Enrolments are now open and brochures are available online at www.ucol.ac.nz or contact Katrina Langdon, K.Langdon@ucol.ac.nz, Ph 06 965 3801 ex 62000.\nThe Whanganui Summer School of the Arts programme includes:\nPainting: R ob McLeod - Marks, multiples and texture, Michael Shepherd - Oil painting, Julie Grieg \xe2\x80\x93 Soft pastel painting.Drawing: Terrie Reddish \xe2\x80\x93 Botanical Drawing.Printmaking: Ron Pokrasso \xe2\x80\x93 Beyond Monotype, Stuart Duffin \xe2\x80\x93 Mezzotint printmaking.Photography: Fleur Wickes \xe2\x80\x93 The New Portrait, Rita Dibert \xe2\x80\x93 Pinholes, Holga\xe2\x80\x99s & Cyanotypes.Sculpture: Brent Sumner \xe2\x80\x93 Darjit Sculpture, Michel Tuffery \xe2\x80\x93 Sculptural Effigy.Glass: Jeff Burnette \xe2\x80\x93 Hot glass techniques, Brock Craig \xe2\x80\x93 Kiln-forming techniques.Jewellery: Craig Winton \xe2\x80\x93 Jewellery Making-Tricks of the trade.Fashion: John Kite \xe2\x80\x93 Pattern drafting for made to measure.Fibre and Fabric: Fiona Wright \xe2\x80\x93 Felting - Text and texture, Deb Price \xe2\x80\x93 Baskets and Beyond.Culinary Arts: Adele Hingston \xe2\x80\x93 Sugar art.Literature: Frankie McMillan \xe2\x80\x93 Creative writing.\nENDS \r \r
'''

@pytest.fixture
def example_fc():
    fc = FeatureCollection()
    fc[u'meta_clean_visible'] = example_text
    return fc

nltk_data_packages = [
    'maxent_treebank_pos_tagger',
    'wordnet',
    'stopwords',
    'punkt',
    'maxent_ne_chunker',
    'words',
]
@pytest.fixture('session')
def nltk_data():
    for data_name in nltk_data_packages:
        print('nltk.download(%r)' % data_name)
        nltk.download(data_name)

def test_entity_names(example_fc, nltk_data):
    '''test for the `entity_names` transform
    '''
    xform = features.entity_names()
    fc = xform.process(example_fc)
    assert u'PERSON' in fc, fc.keys()
    assert u'Craig Winton' in fc[u'PERSON']
    #print(json.dumps(dict(fc), indent=4))
