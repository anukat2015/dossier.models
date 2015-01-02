'''dossier.models.tests provides a fixtures

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2014 Diffeo, Inc.

'''
import pytest

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
