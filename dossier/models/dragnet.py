
from __future__ import division
import argparse
import json
import logging
from math import exp
import operator

import kvlayer
import dblogger
import rejester
import many_stop_words
import yakonfig

stops = many_stop_words.get_stop_words()

try:
    from collections import Counter, defaultdict
except ImportError:
    from backport_collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

from dossier.fc import StringCounter
from dossier.models.web.config import Config
from dossier.models.folder import Folders

logger = logging.getLogger(__name__)

def rejester_run_dragnet(work_unit):
    if 'config' not in work_unit.spec:
        raise rejester.exceptions.ProgrammerError(
            'could not run dragnet without global config')

    web_conf = Config()
    unitconf = work_unit.spec['config']
    with yakonfig.defaulted_config([rejester, kvlayer, dblogger, web_conf],
                                   config=unitconf):

        def make_feature(fc):
            feat = StringCounter()
            rejects = set()
            keepers = set()
            keepers_keys = ['GPE', 'PERSON', 'ORGANIZATION']
            rejects_keys = ['keywords']
            # The features used to pull the keys for the classifier
            for f, strength in [('keywords', 10**4), ('GPE', 1), ('bow', 1), ('bowNP_sip', 10**8), ('bowNP', 10**3), ('PERSON', 10**8), ('ORGANIZATION', 10**6)]:
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

        labels = []
        D = list()
        
        label2fid = dict()

        rejects = set()
        keepers = set()
        # make a classifier target for each *folder*, ignore subfolder structure
        FT = Folders(web_conf.kvlclient)
        for idx, fid in enumerate(FT.folders()):
            label2fid[idx] = fid
            for sid in FT.subfolders(fid):
                for cid, subtopic_id in FT.items(fid, sid):
                    fc = web_conf.store.get(cid)
                    if fc:
                        feat, _rejects, _keepers = make_feature(fc)
                    else:
                        _rejects = {}
                        _keepers = {}
                    D.append(feat)
                    labels.append(idx)
                    rejects.update(_rejects)
                    keepers.update(_keepers)
                    logger.info('observation: %r', cid)

        # Convert the list of Counters into an sklearn compatible format
        logger.info('transforming...')
        v = DictVectorizer(sparse=False)
        X = v.fit_transform(D)
        logger.info('transform fit done.')

        labels = np.array(labels)

        # Fit the sklearn Bernoulli Naive Bayes classifer
        clf = MultinomialNB()
        clf.fit(X, labels)
        logger.info('fit MultinomialNB')

        counts = Counter()
        for cid, fc in web_conf.store.scan():
            feat, _rejects, _keepers = make_feature(fc)
            X = v.transform([feat])
            target = clf.predict(X[0])[0]
            counts[label2fid[target]] += 1

        logger.info('counts done')

        # Extract the learned features that are predictive
        clusters = []
        for idx in sorted(set(labels)):
            logger.info('considering cluster: %d', idx)
            try:
                all_features = v.inverse_transform(clf.feature_log_prob_[idx])[0]
            except: 
                logger.info('beyond edge')
                continue
            words = Counter(all_features)
            ordered = sorted(words.items(), 
                             key=operator.itemgetter(1), reverse=True)
            filtered = [it for it in ordered if (it[0] not in rejects and it[0] in keepers and it[0] not in stops)]
            biggest = exp(filtered[0][1])
            filtered = [(key, int(round(counts[label2fid[idx]] * exp(w) / biggest))) for key, w in filtered]
            logger.info('%s --> %r', label2fid[idx], ['%s: %d' % it for it in filtered[:10]])
            cluster = []
            cluster.append({'caption': label2fid[idx],
                            'weight': counts[label2fid[idx]],
                            'folder_id': None,
                            })
            cluster += [{'caption': caption, 'weight': weight, 'folder_id': label2fid[idx]} for caption, weight in filtered if weight > 0]
            clusters.append(cluster)

        web_conf.kvlclient.setup_namespace({'dragnet': (str,)})
        web_conf.kvlclient.put('dragnet', (('dragnet',), json.dumps({'clusters': clusters})))
        return dict(counts)


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    args = yakonfig.parse_args(p, [kvlayer, yakonfig])

    config = yakonfig.get_global_config()

    class Empty(object): pass
    e = Empty()
    e.spec = dict(config=config)
    rejester_run_dragnet(e)
