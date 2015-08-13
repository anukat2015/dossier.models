'''Detects mention and selector correlations across documents in the
dossiers (folders), thus generating recommendations for querying and
highlighting.

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.

Dragnet
=======
.. autofunction:: make_feature
.. autofunction:: worker

'''
from __future__ import division
import argparse
import json
import logging
from math import exp
import operator
from itertools import islice
import kvlayer
import dblogger
import coordinate
import many_stop_words
import yakonfig
import regex as re

stops = many_stop_words.get_stop_words()

try:
    from collections import Counter, defaultdict
except ImportError:
    from backport_collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

from dossier.fc import StringCounter, FeatureCollection
from dossier.models.web.config import Config
from dossier.models.folder import Folders

logger = logging.getLogger(__name__)


def make_feature(fc):
    '''Builds a new `StringCounter` from the many `StringCounters` in the
    input `fc`.  This StringCounter will define one of the targets for
    the `MultinomialNB` classifier.

    This crucial function decides the relative importance of features
    extracted by the ETL pipeline.  This is essentially a form of
    domain fitting that allows us to tune the extraction to the fields
    that are important to a domain.  However, if the NER for a domain
    is inadequate, then the primary purpose of these relative
    weightings is to remove bogus NER extractions.

    '''
    feat = StringCounter()
    rejects = set()
    keepers = set()
    #keepers_keys = ['GPE', 'PERSON', 'ORGANIZATION', 'usernames']
    keepers_keys = ['phone', 'email'] #['usernames', 'phone', 'email', 'ORGANIZATION', 'PERSON']
    rejects_keys = ['keywords', 'usernames', 'ORGANIZATION', 'PERSON']
    # The features used to pull the keys for the classifier
    for f, strength in [('keywords', 10**4), ('GPE', 1), ('bow', 1), ('bowNP_sip', 10**8),
                        ('phone', 10**12), ('email', 10**12),
                        ('bowNP', 10**3), ('PERSON', 10**8), ('ORGANIZATION', 10**6), ('usernames', 10**12)]:
        if strength == 1:
            feat += fc[f]
        else:
            feat += StringCounter({key: strength * count
                                   for key, count in fc[f].items()})
        if f in rejects_keys:
            map(rejects.add, fc[f])
        if f in keepers_keys:
            map(keepers.add, fc[f])
        if u'' in feat: feat.pop(u'')
    return feat, rejects, keepers


def worker(work_unit, max_sample=1000):
    '''Expects a coordinate WorkUnit for DragNet and runs the following
    steps:

    1. scans all dossiers at the *folder* level and assembles feature
    vectors for each folder -- see `make_feature`

    2. trains a multinomial naive Bayes classifier that treats each
    *folder* as a classifier target.

    3. sample the corpus by scanning up to `max_sample` and applying
    the classifier to each item to get an approx "size" of the Folder

    4. Bootstrap by treating those classifier predictions as truth
    data and extract the learned features that are predictive as new
    query strings.

    5. Put the data in kvlayer for webservice end point to return to
    polling client -- see dossier.models.routes

    '''
    if 'config' not in work_unit.spec:
        raise coordinate.exceptions.ProgrammerError(
            'could not run dragnet without global config')

    web_conf = Config()
    unitconf = work_unit.spec['config']
    with yakonfig.defaulted_config([coordinate, kvlayer, dblogger, web_conf],
                                   config=unitconf):

        labels = []
        D = list()

        label2fid = dict()

        rejects = set()
        keepers = set()

        # 1. make a classifier target for each *folder*, ignoring
        # subfolder structure
        FT = Folders(web_conf.kvlclient)
        for idx, fid in enumerate(FT.folders()):
            label2fid[idx] = fid
            for sid in FT.subfolders(fid):
                for cid, subtopic_id in FT.items(fid, sid):
                    fc = web_conf.store.get(cid)
                    if fc:
                        # NB: first call to make_feature
                        feat, _rejects, _keepers = make_feature(fc)
                    else:
                        _rejects = {}
                        _keepers = {}
                    D.append(feat)
                    labels.append(idx)
                    rejects.update(_rejects)
                    keepers.update(_keepers)
                    logger.info('fid=%r, observation: %r', fid, cid)

        # 2. Convert the StringCounters into an sklearn format and
        # train MultinomialNB
        logger.info('transforming...')
        v = DictVectorizer(sparse=False)
        X = v.fit_transform(D)
        logger.info('transform fit done.')

        labels = np.array(labels)

        # Fit the sklearn Bernoulli Naive Bayes classifer
        clf = MultinomialNB()
        clf.fit(X, labels)
        logger.info('fit MultinomialNB')

        # 3. Scan the corpus up to max_sample putting the items into
        # each target to get an approx "size" of the Folder
        counts = Counter()
        for cid, fc in islice(web_conf.store.scan(), max_sample):
            # build the same feature vector as the training process
            feat, _rejects, _keepers = make_feature(fc)
            X = v.transform([feat])
            # predict which folder it belongs in
            target = clf.predict(X[0])[0]
            # count the effective size of that folder in this sample
            counts[label2fid[target]] += 1

        logger.info('counts done')

        ## 4. Bootstrap by treating those classifier predictions as
        ## truth data and extract the learned features that are
        ## predictive as new query strings.
        clusters = []
        for idx in sorted(set(labels)):
            logger.debug('considering cluster: %d', idx)
            try:
                all_features = v.inverse_transform(clf.feature_log_prob_[idx])[0]
            except:
                logger.warn('beyond edge on cluster %d', idx)
                continue
            words = Counter(all_features)
            ordered = sorted(words.items(),
                             key=operator.itemgetter(1), reverse=True)
            filtered = []
            for it in ordered:
                if is_bad_token(it[0]): continue

                if is_username(it[0]):
                    logger.debug('%r is_username', it[0])
                #else:
                #    continue
                filtered.append(it)
                if len(filtered) > 100: # hard cutoff
                    break

            # normalize cluster size exponentially
            biggest = exp(filtered[0][1])
            # rescale all by biggest
            filtered = [(key, int(round(counts[label2fid[idx]] * exp(w) / biggest))) for key, w in filtered]
            # describe what we just figured out
            logger.info('%s --> %r', label2fid[idx], ['%s: %d' % it for it in filtered[:10]])

            # return build the JSON-serializable format for the
            # DragNet UI embedded inside SortingDesk
            cluster = []
            cluster.append({'caption': label2fid[idx],
                            'weight': counts[label2fid[idx]],
                            'folder_id': None,
                            })
            cluster += [{'caption': caption, 'weight': weight, 'folder_id': label2fid[idx]} for caption, weight in filtered if weight > 0]
            clusters.append(cluster)

        # 5. Put the data in kvlayer for webservice end point to
        # return to polling client
        web_conf.kvlclient.setup_namespace({'dragnet': (str,)})
        web_conf.kvlclient.put('dragnet', (('dragnet',), json.dumps({'clusters': clusters})))
        return dict(counts)


def is_username(s):
    # In the future, we should get username features in this
    #userclf = cyber_text_features.handles.classifier.Classifier('naivebayes')
    #is_username = userclf.classify(s)
    _is_username = (bool(allowed_format_re.match(s)) and bool(has_non_letter_re.search(s))
                   and not has_only_underscore.match(s))
    return _is_username

allowed_format_re = re.compile(ur'^\w(?:\w*(?:[.-_]\w+)?)*(?<=^.{4,32})$')
has_non_letter_re = re.compile(ur'[^a-zA-Z]+')
has_only_underscore = re.compile(ur'^([^a-zA-Z]+_)+[a-zA-Z]*$')
def has_repeating_letter(s):
    for i in range(len(s) - 1):
        if s[i] == s[i+1]: return True
    return False
has_number_re = re.compile(ur'[0-9]')
bad_punctuation_re = re.compile(ur'[&=;"-/]')
def is_bad_token(s):
    if len(s.strip()) == 0: return True
    if bad_punctuation_re.search(s): return True
    return False


def main():
    p = argparse.ArgumentParser()
    args = yakonfig.parse_args(p, [kvlayer, yakonfig])

    config = yakonfig.get_global_config()

    class Empty(object): pass
    e = Empty()
    e.spec = dict(config=config)
    worker(e)


if __name__ == '__main__':
    main()
