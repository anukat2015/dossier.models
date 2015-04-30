from __future__ import absolute_import, division, print_function

import csv
from itertools import ifilter, imap
import json
import sys
import traceback
import urllib

from dossier.fc import FeatureCollection, StringCounter
from dossier.models.etl.interface import ETL, html_to_fc, mk_content_id


class Scrapy(ETL):
    def __init__(self, filelike):
        self.rows = csv.DictReader(filelike)

    def cids_and_fcs(self, limit=5, pool=None):
        mapper = imap if pool is None else pool.imap
        posts = ifilter(lambda d: d['_type'] == 'ForumPostItem', self.rows)
        posts = imap(nice_post, posts)
        return mapper(from_forum_post, posts)


def from_forum_post(row):
    cid = forum_post_id(row)
    try:
        fc = html_to_fc(row['content'].strip(),
                        url=row['thread_link'],
                        timestamp=forum_post_timestamp(row),
                        other_features=forum_post_features(row))
    except:
        fc = None
        print('Could not create FC for %s:' % cid, file=sys.stderr)
        print(traceback.format_exc())
    return cid, fc


def forum_post_features(row):
    fc = FeatureCollection()
    for k in row['author']:
        fc['post_author_' + k] = row['author'][k]
    fc['image_url'] = StringCounter()
    for image_url in row['image_urls']:
        fc['image_url'][image_url] += 1

    others = ['parent_id', 'thread_id', 'thread_link', 'thread_name', 'title']
    for k in others:
        fc['post_' + k] = row[k].decode('utf-8')
    return fc


def forum_post_id(row):
    ticks = forum_post_timestamp(row)
    abs_url = row['thread_link']
    author = row['author'].get('username', 'unknown')
    return mk_content_id('|'.join(map(urlquote, [ticks, abs_url, author])))


def forum_post_timestamp(row):
    return str(int(row['created_at']) / 1000)


def nice_post(row):
    row['author'] = as_json(row['author'])
    row['image_urls'] = row['image_urls'].split(',')
    return row


def as_json(v):
    if v is not None and len(v) > 0:
        try:
            return json.loads(v)
        except ValueError:
            return {}
    return {}


def urlquote(s):
    return urllib.quote(s, safe='~')
