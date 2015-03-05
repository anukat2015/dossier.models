FROM ubuntu:14.04
MAINTAINER Diffeo <support@diffeo.com>

RUN apt-get update -y && apt-get install -y python python-pip

RUN apt-get install -y \
      make libz-dev libxslt1-dev libxml2-dev python-dev \
      python-virtualenv g++ xz-utils gfortran liblzma-dev \
      libpq-dev libfreetype6-dev libblas-dev liblapack-dev \
      libboost-python-dev libsnappy1 libsnappy-dev

RUN apt-get install -y \
      python-numpy python-scipy python-sklearn python-matplotlib \
      python-gevent uwsgi

# Download and install the NLTK corpus.
RUN pip install nltk \
 && python -m nltk.downloader -d /usr/share/nltk_data all

# Now install dossier.models.
RUN pip install --pre dossier.models
RUN pip install gensim

# There is a bug in our pairwise model code that can result in passing a
# NaN to the sklearn model training. In old versions, this causes the model
# to raise an exception. In newer versions, it emits a warning and keeps on
# chugging. So we upgrade! ---AG
RUN pip install --upgrade scikit-learn

ADD config.yaml /config.yaml
ADD background-50000.tfidf.gz /
RUN gunzip /background-50000.tfidf.gz

# Finally, run the dossier.models web server.
EXPOSE 57312
CMD uwsgi \
      --http-socket 0.0.0.0:57312 \
      --wsgi dossier.models.web.wsgi \
      --pyargv "-c /config.yaml" \
      --master \
      --processes 1 \
      --vacuum \
      --harakiri 20 \
      --max-requests 5000
