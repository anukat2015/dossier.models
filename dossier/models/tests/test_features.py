'''``dossier.models.features.basic`` provides simple transforms that
construct features.

.. This software is released under an MIT/X11 open source license.
   Copyright 2012-2015 Diffeo, Inc.

'''
from dossier.fc import StringCounter
import dossier.models.features as features
import pytest


def test_extract_phones():
    txt = '''
Phone: 111-222-3333
Phone: 1112223333
Phone: 1-111-222-3333
Phone: 11112223333
Phone: 222-3333
Phone: 2223333
'''
    assert StringCounter(features.phones(txt)) == StringCounter({
        '1112223333': 2,
        '11112223333': 2,
        '2223333': 2,
    })


def test_a_urls():
    html = '''
<a href="http://ExAmPle.com/My Page.html">
<a href="http://example.com/My%20Page.html">
'''
    assert StringCounter(features.a_urls(html)) == StringCounter({
        'http://example.com/My Page.html': 2,
    })

def test_image_urls():
    html = '''
<img src="http://ExAmPle.com/My Image.jpg">
<img src="http://example.com/My%20Image.jpg">
'''
    assert StringCounter(features.image_urls(html)) == StringCounter({
        'http://example.com/My Image.jpg': 2,
    })


def test_extract_emails():
    txt = '''
email: abc@example.com
email: AbC@eXamPle.com
'''
    assert StringCounter(features.emails(txt)) == StringCounter({
        'abc@example.com': 2,
    })

def test_host_names():
    urls = StringCounter()
    urls['http://www.example.com/folder1'] = 3
    urls['http://www.example.com/folder2'] = 2
    urls['http://www.different.com/folder2'] = 7


    assert features.host_names(urls) == StringCounter({
        'www.example.com': 5,
        'www.different.com': 7,
    })

def test_path_dirs():
    urls = StringCounter()
    urls['http://www.example.com/folder1/folder3/index.html?source=dummy'] = 3
    urls['http://www.example.com/folder2/folder1'] = 2
    urls['http://www.different.com/folder2'] = 7


    assert features.path_dirs(urls) == StringCounter({
        'folder1': 5,
        'folder2': 9,
        'folder3': 3,
        'index.html': 3,
    })


example_usernames_from_paths = [
     (r'http://www.example.com/user/folder3/index.html?source=dummy', 'folder3', 3),
     (r'http://www.example.com/user/myaccount', 'myaccount', 2),
     (r'http://www.different.com/folder3', None, 4),
     (r'http://www.different.com/user/myaccount', 'myaccount', 7),
     (r'http://www.also.com/user', None, 23),
     (r'http://www.also2.com/user/user', 'user', 1),
     (r'http://frob.com/user/my_account/media/Dresses/hi.jpg', 'my_account', 1),
     (r'https://www.facebook.com/my_account', 'my_account', 1),
     (r'https://twitter.com/my_account', 'my_account', 1),
     (r'C:\WINNT\Profiles\myaccount%MyUserProfile%', 'myaccount', 3), # Microsoft Windows NT
     (r'C:\WINNT\Profiles\myaccount', 'myaccount', 3), # Microsoft Windows NT
     (r'd:\WINNT\Profiles\myaccount', 'myaccount', 3), # Microsoft Windows NT
     (r'X:\Documents and Settings\myaccount', 'myaccount', 8), # Microsoft Windows 2000, XP and 2003
     (r'C:\Users\myaccount', 'myaccount', 3), # Microsoft Windows Vista, 7 and 8
     (r'C:\Users\myaccount\dog', 'myaccount', 3), # Microsoft Windows Vista, 7 and 8
     (r'C:\Users\whg\Desktop\Plug\FastGui(LYT)\Shell\Release\Shell.pdb', 'whg', 2),
     (r'C:\Documents and Settings\whg\\Plug\FastGui(LYT)\Shell\Release\Shell.pdb', 'whg', 3),
     (r'C:\Users\whg\Desktop\Plug\FastGui(LYT)\Shell\Release\Shell.pdb', 'whg', 3),
     (r'/home/myaccount$HOME', 'myaccount', 5), # Unix-Based
     (r'/var/users/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/u01/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/user/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/users/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/var/users/myaccount', 'myaccount', 3), # Unix-Derived
     (r'/home/myaccount', 'myaccount', 3), # Linux / BSD (FHS)
     (r'/Users/my_account$HOME', 'my_account', 5), # Mac OS X
     (r'/Users/my_account', 'my_account', 5), # Mac OS X
     (r'/data/media/myaccount', 'myaccount', 5) # Android
     ]

@pytest.mark.parametrize(
    ('url_or_path', 'username', 'count'),
    example_usernames_from_paths
)
def test_usernames(url_or_path, username, count):
    urls = StringCounter()
    urls[url_or_path] += count

    if username is not None:
        results = features.usernames(urls)
        assert results == StringCounter({username: count})
