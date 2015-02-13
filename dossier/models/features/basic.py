from __future__ import absolute_import, division, print_function

from itertools import imap
import logging
import re
import traceback

from bs4 import BeautifulSoup
import urlnorm


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
