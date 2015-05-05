'''``dossier.models.features.basic`` provides simple transforms that
construct features.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2015 Diffeo, Inc.

'''
from __future__ import absolute_import, division, print_function

from itertools import imap
import logging
import regex as re
import traceback

from bs4 import BeautifulSoup
import urlnorm

from urlparse import urlparse
from dossier.fc import StringCounter


logger = logging.getLogger(__name__)


REGEX_PHONES = [
    r'\d-?\d{3}-?\d{3}-?\d{4}',
    r'\d{3}-?\d{3}-?\d{4}',
    r'\d{3}-?\d{4}',
]
REGEX_PHONE = re.compile('|'.join(REGEX_PHONES))

# Be extremely liberal with what we consider an email address.
REGEX_EMAIL = re.compile(r'\b\S+@\S+\.\S+\b', re.IGNORECASE)


def phones(text):
    '''Returns list of phone numbers without punctuation.'''
    return imap(lambda m: m.group(0).replace('-', ''),
                REGEX_PHONE.finditer(text))


def emails(text):
    '''Returns list of phone numbers without punctuation.'''
    return imap(lambda m: m.group(0).lower(), REGEX_EMAIL.finditer(text))


def image_urls(html):
    soup = BeautifulSoup(html)
    for node in soup.find_all('img'):
        try:
            src = node['src']
        except KeyError:
            continue
        yield norm_url(src)


def a_urls(html):
    '''
    return normalized urls found in the 'a' tag
    '''
    soup = BeautifulSoup(html)
    for node in soup.find_all('a'):
        try:
            href = node['href']
        except KeyError:
            continue
        yield norm_url(href)


def host_names(urls):
    '''
    Takes a StringCounter of normalized URL and parses their hostnames

    N.B. this assumes that absolute URLs will begin with

    http://

    in order to accurately resolve the host name.
    Relative URLs will not have host names.
    '''
    host_names = StringCounter()
    for url in urls:
        host_names[urlparse(url).netloc] += urls[url]
    return host_names


def path_dirs(urls):
    '''
    Takes a StringCounter of normalized URL and parses them into
    a list of path directories. The file name is
    included in the path directory list.
    '''
    path_dirs = StringCounter()
    for url in urls:
        for path_dir in filter(None, urlparse(url).path.split('/')):
            path_dirs[path_dir] += urls[url]
    return path_dirs


path_prefixes = r'''user|users|Users|home|data/media|var/users|u01''' + \
                r'''|Documents and Settings|WINNT\\Profiles'''

username_re = re.compile(
    r'^((?P<drive>[A-Za-z]):)?(/|\\)(%s)(/|\\)(?P<username>[^/\\$%%]+)' % path_prefixes
)

def usernames(urls):
    '''Take an iterable of `urls` of normalized URL or file paths and
    attempt to extract usernames.  Returns a list.

    '''
    usernames = StringCounter()
    for url, count in urls.items():
        uparse = urlparse(url)
        path = uparse.path
        hostname = uparse.hostname
        m = username_re.match(path)
        if m:
            usernames[m.group('username')] += count
        elif hostname in ['twitter.com', 'www.facebook.com']:
            usernames[path.lstrip('/')] += count
    return usernames


def norm_url(url):
    url = uni(url).encode('utf-8')
    try:
        return urlnorm.norm(url)
    except urlnorm.InvalidUrl:
        # Happens when the URL is relative. Call path normalization directly.
        return urlnorm.norm_path('', url)
    except:
        traceback.print_exc()
    return None


def uni(s):
    if not isinstance(s, unicode):
        try:
            return s.decode('utf-8')
        except UnicodeDecodeError:
            return s.decode('latin-1')
    return s
