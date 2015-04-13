'''
Feature extraction
==================
``dossier.models.features`` provides some convenience functions for
feature extraction.

.. autofunction:: noun_phrases
'''
from dossier.models.features.basic import emails, image_urls, phones, a_urls
from dossier.models.features.basic import host_names, path_dirs
from dossier.models.features.sip import noun_phrases, sip_noun_phrases
from dossier.models.features.stopwords import stopwords


__all__ = [
    'emails', 'image_urls', 'a_urls', 'phones',
    'noun_phrases', 'sip_noun_phrases',
    'stopwords', 'host_names', 'path_dirs'
]
