'''
Create a keyword searches from an entity profile.
'''
from __future__ import absolute_import, division
import json
import logging
import os
import sys
import time
import datetime

import argparse
import dblogger
import json

import yakonfig
import streamcorpus
from streamcorpus import Chunk, make_stream_item
import streamcorpus_pipeline
from streamcorpus_pipeline.stages import PipelineStages
from streamcorpus_pipeline._pipeline import PipelineFactory

from dossier.fc import FeatureCollectionChunk as FCChunk
from dossier.fc import FeatureCollection, StringCounter


import operator
from collections import Counter

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB

logger = logging.getLogger(__name__)

'''
Given a labeled collection of feature collections, 
train a classifier to identify the entity using the simplest
off the shelf tools, such as scikit-learn Bernoulli naive Bayes:

http://scikit-learn.org/stable/modules/naive_bayes.html

http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes
.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
'''

def extract(positive_fcs, negative_fcs, features=None):
    '''Takes a labeled set of feature collections (positive and negative)
       and the features wanted. And trains a Naive Bayes classifier on
       the underlying keys of the set of selected features features.
       If no features are selected, all are used.

       Returns two Counters of keywords weighted by strength. The
       first are feature keys that were predictive of the positive
       label and the second are the feature keys are were predictive
       of the negative label. 

    `*_fcs' is the list of feature collections, positive label and
            negative label respectively.

    `features' designates which specific feature gets vectorized the
               other features are ignored.

    '''
    
    # Vector of labels
    labels = np.array([1] * len(positive_fcs) + [0] * len(negative_fcs))

    # Used to convert the feature collection keys into a sklearn
    # compatible format
    v = DictVectorizer(sparse=False)

    D = list()
    for fc in (positive_fcs + negative_fcs):
        feat = StringCounter()

        # The features used to pull the keys for the classifier
        for f in features:
            feat += fc[f]

        D.append(feat)

    # Convert the list of Counters into an sklearn compatible format
    X = v.fit_transform(D)

    # Fit the sklearn Bernoulli Naive Bayes classifer
    clf = BernoulliNB()
    clf.fit(X, labels)

    # Extract the learned features that are predictive of the positive
    # and negative class
    positive_keywords = v.inverse_transform(clf.feature_log_prob_[1])[0]
    negative_keywords = v.inverse_transform(clf.feature_log_prob_[0])[0]
    
    return Counter(positive_keywords), Counter(negative_keywords)

if __name__ == '__main__':
    '''This can be used to test this code on features collections created
    by the treelab pipeline
    '''
    
    parser = argparse.ArgumentParser(
        usage='python extractor.py corpus.fc',
        description=__doc__)
    parser.add_argument('corpus', 
        help='path feature collection chunk that contains the corpus')
    args = parser.parse_args()

    positive_fcs = list()
    negative_fcs = list()

    # Build a fake labeled corpus
    for i, fc in enumerate(FCChunk(args.corpus)):
        if i % 2:
            positive_fcs.append(fc)
        else:
            negative_fcs.append(fc)

    keywords = extract(
        positive_fcs,
        negative_fcs,
        features = ['both_bow_3', 
                    'both_con_3', 
                    'both_co_LOC_3', 
                    'both_co_ORG_3']
    )
    
    print 'Predictive of positive labels:'
    print keywords[0]

    print 'Predictive of positive labels:'
    print keywords[1]
    
