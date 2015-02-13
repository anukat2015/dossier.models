from dossier.fc import StringCounter
import dossier.models.features as features


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


def test_image_urls():
    html = '''
<img src="http://ExAmPle.com/My Image.jpg">
<img src="http://example.com/My%20Image.jpg">
'''
    assert StringCounter(features.image_urls(html)) == StringCounter({
        'http://example.com/My%20Image.jpg': 2,
    })


def test_extract_emails():
    txt = '''
email: abc@example.com
email: AbC@eXamPle.com
'''
    assert StringCounter(features.emails(txt)) == StringCounter({
        'abc@example.com': 2,
    })
