'''Provides tools for working with Google's Custom Search API

Documentation here:  https://developers.google.com/custom-search/docs/start

Note that the Google Custom Site Search interface has quite a bit of
granularity.  You can use lists of regex patterns to specify specific
sites, subdirectories, pages, etc

Pricing for the API: https://support.google.com/customsearch/answer/72334?hl=en

(starts at half penny per query)
'''
import gzip
import json
import logging
import StringIO
import sys
import time
import traceback
import urllib
import urllib2

logger = logging.getLogger(__name__)

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) ' \
             'AppleWebKit/535.1 (KHTML, like Gecko) ' \
             'Chrome/14.0.835.202 Safari/535.1'
RESULTS_PER_PAGE = 10


class Google(object):
    url = "https://www.googleapis.com/customsearch/v1" \
          "?key=%(key)s&cx=008142435548003360103:a-_62xajpao" \
          "&q=%(query)s&num=%(num)s&safe=off&start=%(start)s"

    #012511506682437389639:sw8_ywndm-g
    #


    def __init__(self, api_key):
        self.api_key = api_key

    def web_search_with_paging(self, query, limit=None):
        start = 1
        while 1:
            results = self.web_search(query, start=start)
            if 'items' not in results:
                break
            for item in results['items']:
                yield item
                start += 1
            if start == results.get("searchInformation").get("totalResults"):
                break
            if limit is not None and start + RESULTS_PER_PAGE >= limit:
                break

    def web_search(self, query, start=0, limit=100, max_tries=3):
        '''
        encapsulates urllib retrieval for fetching JSON results from
        Google's Custom Search API.  Returns a deserialized result set.
        '''
        tries = 0
        if isinstance(query, unicode):
            query = query.encode('utf8')
        url = self.url % dict(key=self.api_key,
                              query=urllib.quote(query.strip()),
                              num=min(10, limit - start),
                              start=start)
        logger.info("fetching: %s" % url)
        while 1:
            try:
                request = urllib2.Request(url)
                request.add_header('Accept-encoding', 'gzip')
                # We do we set this?  Remnant from pre-API version?
                request.add_header('User-Agent', USER_AGENT)
                opener = urllib2.build_opener()

                fh = opener.open(request, timeout=60)
                data = fh.read()
                if fh.headers.get('Content-Encoding') == 'gzip':
                    compressedstream = StringIO.StringIO(data)
                    fh = gzip.GzipFile(fileobj=compressedstream)
                    data = fh.read()
                return json.loads(data)
            except Exception, exc:
                logger.info(traceback.format_exc(exc))
                if tries >= max_tries:
                    sys.exit("failed %d times to fetch %s" % (max_tries, url))
                else:
                    logger.info("failed to fetch\n\t%s\nwill try "
                                "%d more times" % (url, max_tries - tries))
                    tries += 1
                    time.sleep(2 ** tries)
