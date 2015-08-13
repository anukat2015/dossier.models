
import argparse
import cbor
import dblogger
import kvlayer
import yakonfig

from .worker import traverse_extract_fetch
from dossier.models.web.config import Config


def main():
    p = argparse.ArgumentParser('simple debugging tool for watching the linker and OpenQuery')
    p.add_argument('action', help='either `run` or `cache` or `delete`')
    p.add_argument('folder', help='folder name')
    p.add_argument('subfolder', help='subfolder name')
    args = yakonfig.parse_args(p, [kvlayer, yakonfig])

    config = yakonfig.get_global_config()

    key = cbor.dumps((args.folder.replace(' ', '_'), args.subfolder.replace(' ', '_')))

    if args.action == 'run':
        web_conf = Config()
        with yakonfig.defaulted_config([kvlayer, dblogger, web_conf], config=config):
            traverse_extract_fetch(web_conf, key, stop_after_extraction=True)

    elif args.action == 'delete':
        kvlclient = kvlayer.client()
        kvlclient.setup_namespace({'openquery': (str,)})
        kvlclient.delete('openquery', (key,))
        print('deleted %r' % key)

    elif args.action == 'cache':

        kvlclient = kvlayer.client()
        kvlclient.setup_namespace({'openquery': (str,)})
        count = 0
        for rec in kvlclient.scan('openquery'):
            count += 1
            if rec[0][0] == key:
                print rec
        print('%d cached queries' % count)
