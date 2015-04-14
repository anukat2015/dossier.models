'''``dossier.models.features.basic`` provides simple transforms that
construct features.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2015 Diffeo, Inc.

'''
from __future__ import absolute_import, division, print_function

from itertools import imap
import logging
import re
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
        try:
            yield urlnorm.norm(src)
        except urlnorm.InvalidUrl:
            # Happens when the URL is relative. Call path normalization
            # directly.
            yield urlnorm.norm_path('', src)
        except:
            traceback.print_exc()
            continue

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
        try:
            yield urlnorm.norm(href)
        except urlnorm.InvalidUrl:
            # Happens when the URL is relative. Call path normalization
            # directly.
            yield urlnorm.norm_path('', href)
        except:
            traceback.print_exc()
            continue

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

def usernames(urls):
    '''
    Takes a StringCounter of normalized URL and attempts
    to extract a username.
    '''
    usernames = StringCounter()
    for url in urls:
        get_next = False
        for path_dir in filter(None, urlparse(url).path.split('/')):
            if get_next:
                get_next = False
                usernames[path_dir] += urls[url]
            if path_dir == 'user':
                get_next = True
    return usernames
