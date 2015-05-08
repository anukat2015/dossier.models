from __future__ import absolute_import, division, print_function

import argparse
from itertools import chain, islice
import multiprocessing
from streamcorpus_pipeline._clean_visible import cleanse, make_clean_visible
import sys
import zlib

import cbor
from gensim import corpora, models
import happybase

import dossier.models.features as features
from dossier.web import streaming_sample
import kvlayer
import yakonfig


def status(*args, **kwargs):
    kwargs['end'] = ''
    args = list(args)
    args[0] = '\033[2K\r' + args[0]
    print(*args, **kwargs)
    sys.stdout.flush()


def batch_iter(n, iterable):
    iterable = iter(iterable)
    while True:
        yield chain([next(iterable)], islice(iterable, n-1))


def unpack_noun_phrases(row):
    body = cbor.loads(zlib.decompress(row['f:response.body']))
    body = make_clean_visible(body.encode('utf-8')).decode('utf-8')
    body = cleanse(body)
    return features.noun_phrases(body)


class App(yakonfig.cmd.ArgParseCmd):
    def __init__(self, *args, **kwargs):
        yakonfig.cmd.ArgParseCmd.__init__(self, *args, **kwargs)

    def args_tfidf(self, p):
        p.add_argument('--host', default='localhost')
        p.add_argument('--port', default=9090, type=int)
        p.add_argument('--table-prefix', default='')
        p.add_argument('--limit', default=100, type=int)
        p.add_argument('--batch-size', default=1000, type=int)
        p.add_argument('-p', '--processes',
                       default=multiprocessing.cpu_count(), type=int)
        p.add_argument('ids', metavar='INPUT_ROW_KEY_SAMPLE_FILE',
                       help='A file containing row keys to use for a sample.')
        p.add_argument('out', metavar='OUTPUT_TFIDF_MODEL_FILE',
                       help='The file path to write the tfidf model to.')

    def do_tfidf(self, args):
        conn = happybase.Connection(host=args.host, port=args.port,
                                    table_prefix=args.table_prefix)
        t = conn.table('artifact')
        corpus = []
        print('Extracting random sample...')
        sample = streaming_sample(open(args.ids), args.limit)

        print('Building corpus...')
        batches = batch_iter(args.batch_size, (s.strip() for s in sample))
        pool = multiprocessing.Pool(processes=args.processes)
        for i, batch in enumerate(batches, 1):
            rows = (row for _, row in t.rows(list(batch)))
            for noun_phrases in pool.imap(unpack_noun_phrases, rows):
                corpus.append(noun_phrases)
            status('%d of %d batches done' % (i, args.limit / args.batch_size))

        print('Computing model...')
        dictionary = corpora.Dictionary(corpus)
        bows = [dictionary.doc2bow(tokens) for tokens in corpus]
        tfidf = models.TfidfModel(bows, id2word=dictionary)
        tfidf.save(args.out)

    def args_ids(self, p):
        p.add_argument('--host', default='localhost')
        p.add_argument('--port', default=9090, type=int)
        p.add_argument('--table-prefix', default='')
        p.add_argument('--limit', default=None, type=int)

    def do_ids(self, args):
        conn = happybase.Connection(host=args.host, port=args.port,
                                    table_prefix=args.table_prefix)
        t = conn.table('artifact')
        hbase_filter = 'FirstKeyOnlyFilter() AND KeyOnlyFilter()'
        ids = islice(enumerate(t.scan(filter=hbase_filter)), args.limit)
        for i, (key, data) in ids:
            print(key)
            if i % 100000 == 0:
                print('%d keys received' % i, file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        description='Specific utilities for working with the ad corpus.')
    app = App()
    app.add_arguments(p)
    args = yakonfig.parse_args(p, [kvlayer, yakonfig])
    app.main(args)


if __name__ == '__main__':
    main()
