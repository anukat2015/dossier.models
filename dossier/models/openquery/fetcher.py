import logging
import os
import time

import dateutil.parser
import requests
import streamcorpus
import trollius as asyncio

logger = logging.getLogger(__name__)


class Fetcher(object):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/38.0.2125.122 Safari/537.36',
        'Referer': 'https://www.google.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,'
        'image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip',
        'Accept-Language': 'en-US,en;q=0.8',
    }

    def __init__(self, timeout=60):
        self.session = requests.Session()
        self.timeout = timeout

    def get_async(self, urls, callback):
        def get(url, headers):
            resp = requests.get(url, headers=headers)
            if resp:
                si = self.process_response(resp)
                callback(si, resp.url)

        @asyncio.coroutine
        def run(urls, concurrency, loop):
            urls = list(urls)
            pending = [loop.run_in_executor(None, get, url, self.headers) for url in urls[:concurrency]]
            rest = urls[concurrency:]
            while pending:
                done, pending = yield asyncio.From(asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED))
                while rest and len(pending) < concurrency:
                    pending.add(loop.run_in_executor(None, get, rest.pop(), self.headers))
                for future in done:
                    try:
                        si = future.result()
                        # callback(si, si.original_url)
                    except Exception:
                        logger.info('failed on url', exc_info=True)

        loop = asyncio.get_event_loop()
        # TODO: make magic concurrency number a parameter
        loop.run_until_complete(run(urls, concurrency=32, loop=loop)) 

    def get(self, url):
        req = requests.Request('GET', url, headers=self.headers)

        try:
            prepped = self.session.prepare_request(req)
            resp = self.session.send(prepped, timeout=self.timeout)
        except Exception:
            logger.info('failed on %r', url, exc_info=True)
            return None

        return self.process_response(resp, url)

    def process_response(self, resp):
        logger.info('retrieved %d bytes for %r', len(resp.content), resp.url)

        last_modified = resp.headers.get('last-modified')
        if last_modified:
            try:
                last_modified = dateutil.parser.parse(last_modified)
                last_modified = int(last_modified.strftime('%s'))
            except Exception:
                last_modified = None
        if not last_modified:
            last_modified = int(time.time())
        si = streamcorpus.make_stream_item(
            last_modified,
            resp.url
            )
        # don't try to convert it... e.g. if we got a PDF
        si.original_url = resp.url
        si.body.raw = resp.content
        media_type = resp.headers.get('content-type')
        try:
            media_type = (media_type
                          .decode('utf8', 'ignore')
                          .encode('utf8', 'ignore'))
        except Exception:
            media_type = repr(media_type)
        si.body.media_type = media_type
        si.body.encoding = resp.apparent_encoding

        return si


class ChunkRoller(object):
    def __init__(self, chunk_dir, chunk_max=500):
        self.chunk_dir = chunk_dir
        self.chunk_max = chunk_max
        self.t_path = os.path.join(chunk_dir, 'tmp.sc.xz')
        self.o_chunk = None

    def add(self, si):
        '''puts `si` into the currently open chunk, which it creates if
        necessary.  If this item causes the chunk to cross chunk_max,
        then the chunk closed after adding.

        '''
        if self.o_chunk is None:
            if os.path.exists(self.t_path):
                os.remove(self.t_path)
            self.o_chunk = streamcorpus.Chunk(self.t_path, mode='wb')
        self.o_chunk.add(si)
        logger.debug('added %d-th item to chunk', len(self.o_chunk))
        if len(self.o_chunk) == self.chunk_max:
            self.close()

    def close(self):
        if self.o_chunk:
            self.o_chunk.close()
            o_path = os.path.join(
                self.chunk_dir,
                '%d-%s.sc.xz' % (len(self.o_chunk), self.o_chunk.md5_hexdigest)
            )
            os.rename(self.t_path, o_path)
            self.o_chunk = None
            logger.info('rolled chunk to %s', o_path)
