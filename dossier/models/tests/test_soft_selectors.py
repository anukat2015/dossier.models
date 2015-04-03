

from dossier.models.soft_selectors import make_ngram_corpus

def test_make_ngram_corpus():
    corpus_clean_visibles = ['''
Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture 
''',
'''
Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture 
''',
'''
Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture 
''',
'''
Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture 
''',
'''
Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture Enlarge Picture 
''',
'''
to decipher a hastily scribbled note
''',
'''
to decipher a hastily scribbled note
''',
'''
to decipher a hastily scribbled note
''',
'''
to decipher a hastily scribbled note
''',
'''
to decipher a hastily scribbled note
''',
]
    candidates = make_ngram_corpus(corpus_clean_visibles, 6, True)

    count = 0
    for c in candidates:
        if not c: continue
        assert c == ['to decipher a hastily scribbled note']
        count += 1

    assert count == 5
