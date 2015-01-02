'''
.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2014 Diffeo, Inc.

``dossier.models`` provides search engines and a :mod:`dossier.web`
application for working with active learning.

.. autofunction:: similar
.. autofunction:: dissimilar
.. autoclass:: PairwiseFeatureLearner

'''
from dossier.models import features
from dossier.models.pairwise import PairwiseFeatureLearner, similar, dissimilar

__all__ = [
    'PairwiseFeatureLearner', 'similar', 'dissimilar',
    'features',
]
