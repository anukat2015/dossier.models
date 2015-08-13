'''
.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2014 Diffeo, Inc.

``dossier.models`` provides search engines and a :mod:`dossier.web`
application for working with active learning.

.. automodule:: dossier.models.web.run
.. automodule:: dossier.models.pairwise
.. automodule:: dossier.models.features
.. automodule:: dossier.models.etl
.. automodule:: dossier.models.dragnet
.. automodule:: dossier.models.soft_selectors
.. automodule:: dossier.models.linker
'''
from dossier.models import features
from dossier.models.pairwise import PairwiseFeatureLearner, similar, dissimilar

__all__ = [
    'PairwiseFeatureLearner', 'similar', 'dissimilar',
    'features',
]
