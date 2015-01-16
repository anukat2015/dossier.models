FROM ubuntu:14.04
MAINTAINER Diffeo <support@diffeo.com>

RUN apt-get update -y \
 && apt-get install -y python
 && easy_install pip
 && pip install --upgrade pip
 && pip install --upgrade setuptools

RUN apt-get install -y \
      make libz-dev libxslt1-dev libxml2-dev python-dev \
      python-virtualenv g++ xz-utils gfortran liblzma-dev \
      libpq-dev libfreetype6-dev libblas-dev liblapack-dev \
      libboost-python-dev

RUN apt-get install -y \
      python-numpy python-scipy python-sklearn python-matplotlib \
      python-gevent uwsgi

# Download and install the NLTK corpus.
RUN pip install nltk \
 && python -m nltk.downloader -d /usr/share/nltk_data all

RUN apt-get install -y libsnappy1 libsnappy-dev

# Now install dossier.models.
RUN pip install --pre dossier.models

ADD config.yaml /config/config.yaml
ADD demo.db /kvlayer/demo.db

# Finally, run the dossier.models web server.
EXPOSE 57312
CMD uwsgi \
      --http-socket 0.0.0.0:57312 \
      --wsgi dossier.models.web.wsgi \
      --pyargv "-c /config/config.yaml" \
      --master \
      --processes 1 \
      --vacuum \
      --harakiri 20 \
      --max-requests 5000
