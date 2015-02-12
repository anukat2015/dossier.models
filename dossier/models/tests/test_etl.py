from dossier.fc import StringCounter
from dossier.models.etl import extract_phones


def test_extract_phones():
    txt = '''
Phone: 111-222-3333
Phone: 1112223333
Phone: 1-111-222-3333
Phone: 11112223333
Phone: 222-3333
Phone: 2223333
'''
    assert StringCounter(extract_phones(txt)) == StringCounter({
        '1112223333': 2,
        '11112223333': 2,
        '2223333': 2,
    })
