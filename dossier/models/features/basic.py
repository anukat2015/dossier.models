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
    r'\d{2}-?\d{11}',
    r'\d{2,3}-?\d{7,15}',
]
REGEX_PHONE = re.compile('|'.join(REGEX_PHONES))

# Be extremely liberal with what we consider an email address.
REGEX_EMAIL = re.compile(r'\b\S+@\S+\.\S+\b', re.IGNORECASE)

REGEX_SKYPE = re.compile(ur'skype(\s|ID|handle|name|\:)*'
                         ur'(?P<skype>[a-zA-Z][a-zA-Z0-9,-._]{5,31})',
                         flags = re.UNICODE | re.IGNORECASE)

def phones(text):
    '''Returns list of phone numbers without punctuation.'''
    return imap(lambda m: m.group(0).replace('-', ''),
                REGEX_PHONE.finditer(text))


def emails(text):
    '''Returns list of phone numbers without punctuation.'''
    return imap(lambda m: m.group(0).lower(), REGEX_EMAIL.finditer(text))

def skypes(text):
    '''Returns list of skype handles.'''
    return imap(lambda m: m.group('skype'), REGEX_SKYPE.finditer(text))

def image_urls(html):
    soup = BeautifulSoup(html, "lxml")
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
    soup = BeautifulSoup(html, 'lxml')
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


def norm_url(url):
    url = uni(url).encode('utf-8')
    try:
        return urlnorm.norm(url)
    except urlnorm.InvalidUrl:
        # Happens when the URL is relative. Call path normalization directly.
        try:
            return urlnorm.norm_path('', url)
        except UnicodeDecodeError:
            return url

    except UnicodeDecodeError:
        # work around for bug in urlnorm on unicode url
        return url
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




REGEX_ICQ = re.compile(ur'icq(\s|number|\:)*(?P<icq>[0-9-]{3,9}(\s*[0-9-]{,9})*)',
                       flags = re.UNICODE | re.IGNORECASE)

REGEX_DIGITS = re.compile(r'[^0-9]', flags=re.MULTILINE)

def ICQs(text):
    ## TODO: remove trailing whitespace from RAW

    icqs = []
    for match in REGEX_ICQ.finditer(text):
        sub = match.span('icq')
        raw_string = text[sub[0]:sub[1]]
        icq = digits_re.sub('', raw_string)
        if icq:
            try:
                icq = int(icq)
            except:
                continue
            icqs.append(str(icq))
    return icqs
