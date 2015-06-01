'''Named entity extractor used to create features in FCs.

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.

'''
from __future__ import absolute_import, division, print_function

try:
    from collections import defaultdict
except ImportError:
    from backport_collections import defaultdict

import nltk

from dossier.fc import StringCounter
from streamcorpus_pipeline import cleanse

class entity_names(object):
    '''Transform on :class:`~dosser.fc.FeatureCollection` that constructs
    `StringCounter` features using `nltk`-based NER.

    '''
    default_config = {
        'text_source': u'meta_clean_visible',
    }

    def __init__(self, config=None):
        if config is None:
            self.config = self.default_config
        else:
            self.config = config

    def process(self, fc, context=None):
        text_source = self.config.get('text_source')
        if text_source and text_source in fc:
            text = fc[text_source]
        else:
            return fc
        names = defaultdict(StringCounter)
        for sent in nltk.sent_tokenize(text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label'):
                    label = chunk.label()
                    name = ' '.join(c[0] for c in chunk.leaves())
                    if not isinstance(name, unicode):
                        name = unicode(name, 'utf-8')
                    name = cleanse(name)
                    #print chunk.node, name
                    names[label][name] += 1
        for entity_type, name_counts in names.items():
            fc[entity_type] = name_counts
        return fc
