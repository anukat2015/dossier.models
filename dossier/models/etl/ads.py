from __future__ import absolute_import, division, print_function

from collections import defaultdict
import re
import traceback
import sys
import zlib

import cbor
import happybase
import happybase.hbase.ttypes as ttypes

from dossier.models.etl.interface import ETL, html_to_fc, mk_content_id
from dossier.fc import StringCounter


class Ads(ETL):
    def __init__(self, host, port, table_prefix=''):
        self.host = host
        self.port = port
        self.table_prefix = table_prefix
        self.last_key = None

    def get_conn(self):
        return happybase.Connection(host=self.host, port=self.port,
                                    table_prefix=self.table_prefix)

    def cids_and_fcs(self, mapper, start, end, limit=5):
        return mapper(row_to_content_obj,
                      self.get_artifact_rows(limit=limit,
                                             start_key=start, stop_key=end))

    def get_artifact_rows(self, limit=5, start_key=None, stop_key=None):
        while True:
            try:
                scanner = self.get_scanner(limit=limit,
                                           start_key=start_key,
                                           stop_key=stop_key)
                for key, data in scanner:
                    if key == self.last_key:
                        # Don't re-process troublesome keys.
                        continue
                    self.last_key = key
                    yield key, unpack_artifact_row(data)
                break
            except ttypes.IOError:
                # Die HBase, die!
                continue

    def get_scanner(self, limit=5, start_key=None, stop_key=None):
        if self.last_key is not None:
            start_key = self.last_key
        t = self.get_conn().table('artifact')
        return t.scan(row_start=start_key, row_stop=stop_key,
                      limit=limit, batch_size=20)


def row_to_content_obj(key_row):
    '''Returns ``FeatureCollection`` given an HBase artifact row.

    Note that the FC returned has a Unicode feature ``artifact_id``
    set to the row's key.
    '''
    key, row = key_row
    cid = mk_content_id(key.encode('utf-8'))
    response = row.get('response', {})

    other_bows = defaultdict(StringCounter)
    for attr, val in row.get('indices', []):
        other_bows[attr][val] += 1
    try:
        artifact_id = key
        if isinstance(artifact_id, str):
            artifact_id = unicode(artifact_id, 'utf-8')
        fc = html_to_fc(
            response.get('body', ''),
            url=row.get('url'), timestamp=row.get('timestamp'),
            other_features=dict(other_bows, **{'artifact_id': artifact_id}))
    except:
        fc = None
        print('Could not create FC for %s:' % cid, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
    return cid, fc


def unpack_artifact_row(row):
    data = {
        'url': row['f:url'],
        'timestamp': int(row['f:timestamp']),
        'request': {
            'method': row['f:request.method'],
            'client': cbor.loads(zlib.decompress(row['f:request.client'])),
            'headers': cbor.loads(zlib.decompress(row['f:request.headers'])),
            'body': cbor.loads(zlib.decompress(row['f:request.body'])),
        },
        'response': {
            'status': row['f:response.status'],
            'server': {
                'hostname': row['f:response.server.hostname'],
                'address': row['f:response.server.address'],
            },
            'headers': cbor.loads(zlib.decompress(row['f:response.headers'])),
            'body': cbor.loads(zlib.decompress(row['f:response.body'])),
        },
        'indices': [],
    }
    for kk, vv in row.items():
        mm = re.match(r"^f:index\.(?P<key>.*)\.[0-9]+$", kk)
        if mm is not None:
            data['indices'].append((mm.group('key'), vv))
    return data
