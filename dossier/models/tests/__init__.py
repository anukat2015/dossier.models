'''dossier.models.tests provides a fixtures

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2015 Diffeo, Inc.

'''
import pytest

import nltk

from dossier.fc import FeatureCollection
from dossier.label import LabelStore
from dossier.store import Store
import kvlayer

@pytest.yield_fixture
def kvl():
    config = {
        'storage_type': 'local',
        'app_name': 'diffeo',
        'namespace': 'dossier.models.tests',
    }
    client = kvlayer.client(config)
    yield client
    client.close()


@pytest.yield_fixture
def store(kvl):
    client = Store(kvl, feature_indexes=[u'feature'])
    yield client
    client.delete_all()


@pytest.yield_fixture
def label_store(kvl):
    client = LabelStore(kvl)
    yield client
    client.delete_all()


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
