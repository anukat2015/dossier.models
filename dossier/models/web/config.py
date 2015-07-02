from __future__ import absolute_import, division, print_function

from gensim import models

from dossier.models.openquery.google import Google
import dossier.web as web


class Config(web.Config):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self._tfidf = None

    @property
    def config_name(self):
        return 'dossier.models'

    def normalize_config(self, config):
        super(Config, self).normalize_config(config)
        try:
            tfidf_path = self.config['tfidf_path']
        except KeyError:
            self._tfidf = False  # service available but absent
        else:
            self._tfidf = models.TfidfModel.load(tfidf_path)

    @property
    def tfidf(self):
        return self._tfidf

    @property
    def google(self):
        api_key = self.config.get('google_api_search_key')
        if api_key is None:
            return None
        return Google(api_key)
