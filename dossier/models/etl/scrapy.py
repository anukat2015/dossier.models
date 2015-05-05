from __future__ import absolute_import, division, print_function

from itertools import ifilter, imap, islice
import json
import re
import sys
import traceback
import urllib
from urlparse import urljoin, urlparse, urlunparse

import urlnorm

from dossier.fc import FeatureCollection, StringCounter
from dossier.models.etl.interface import ETL, html_to_fc, mk_content_id


class Scrapy(ETL):
    @staticmethod
    def detect_url_prefix(filelike):
        # This is a heuristic based on my observation of the data. This assumes
        # that all data is coming from a single domain and that all discovered
        # relative links have exactly the same prefix. I'm dubious. ---AG
        #
        # The heuristic is to find an absolute URL and a relative URL, then
        # discover the prefix.
        #
        # If one could not be detected `None` is returned.
        def find_prefix():
            for uabs in uabss:
                for urel in urels:
                    prefix = prefix_of(uabs, urel)
                    if prefix is not None:
                        return prefix
            return None

        def prefix_of(uabs, urel):
            if len(uabs) >= len(urel) and uabs[-len(urel):] == urel:
                return uabs[:-len(urel)]
            return None

        def add_url(url):
            if url is None:
                return
            url = urlunparse(urlparse(url)._replace(query=''))
            if re.search('^http', url):
                uabss.add(url)
            else:
                urels.add(url.lstrip('.'))

        uabss, urels = set(), set()
        for row in ifilter(None, imap(json_maybe_loads, filelike)):
            if row.get('_type') == 'ForumPostItem':
                add_url(row.get('thread_link'))
                add_url(row.get('author', {}).get('link'))
            elif row.get('_type') == 'CcaItem':
                add_url(row.get('url'))
            prefix = find_prefix()
            if prefix is not None:
                return prefix
        return None

    def __init__(self, filelike, url_prefix=None):
        self.rows = ifilter(None, imap(json_maybe_loads, filelike))
        self.url_prefix = url_prefix

    def cids_and_fcs(self, mapper, limit=5):
        posts = ifilter(lambda d: d.get('_type') == 'ForumPostItem', self.rows)
        return mapper(from_forum_post,
                      islice(imap(self.sanitize, posts), limit))

    def sanitize(self, post):
        for n in ['link', 'avatar']:
            if n not in post['author']:
                continue
            post['author'][n] = self.sanitize_url(post['author'][n])
        if 'thread_link' in post:
            post['thread_link'] = self.sanitize_url(post['thread_link'])
        return post

    def sanitize_url(self, url):
        url = normurl(url)
        if self.url_prefix is None:
            return url
        if re.search('^http', url):
            return url
        return normurl(urljoin(self.url_prefix, url))


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

    if 'image_urls' in row:
        fc['image_url'] = StringCounter()
        for image_url in row['image_urls']:
            fc['image_url'][image_url] += 1

    others = ['parent_id', 'thread_id', 'thread_link', 'thread_name', 'title']
    for k in others:
        if k in row:
            fc['post_' + k] = uni(row[k])
    return fc


def forum_post_id(row):
    ticks = forum_post_timestamp(row)
    abs_url = row['thread_link']
    author = row['author'].get('username', 'unknown')
    return mk_content_id('|'.join(map(urlquote, [ticks, abs_url, author])))


def forum_post_timestamp(row):
    return str(int(row['created_at']) / 1000)


def urlquote(s):
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    return urllib.quote(s, safe='~')


def normurl(url):
    try:
        return urlnorm.norm(url)
    except urlnorm.InvalidUrl:
        return urlnorm.norm_path('', url)


def uni(s):
    if isinstance(s, str):
        return unicode(s, 'utf-8')
    return s


def json_maybe_loads(s):
    try:
        d = json.loads(s)
    except:
        return None
    if 'thread_link' not in d:
        return None
    return d
