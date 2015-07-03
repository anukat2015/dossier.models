'''test of NP and SIP extraction

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.

'''
from __future__ import absolute_import, division, print_function

from dossier.models.features.sip import sip_noun_phrases, noun_phrases

def test_noun_phrases():

    text = '''
This is a test of noun phrase extraction on New York Harbour and the cheese burger!
'''
    np = noun_phrases(text)

    assert 'chees_burger' in np
    
