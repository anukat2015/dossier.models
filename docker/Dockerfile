FROM centos:centos6
MAINTAINER Diffeo <support@diffeo.com>

RUN yum install -y epel-release \
 && yum update -y \
 && yum clean all

# Install C libraries needed for future Python installs;
# Update the system to current packages;
# Get a python27-based virtual environment
RUN yum install -y centos-release-SCL \
 && yum install -y \
      python27-python-virtualenv gcc-c++ gcc-gfortran git make pkgconfig \
      redhat-rpm-config rpm-build yum-utils wget s3cmd tar zlib-devel \
      bzip2-devel lapack-devel blas-devel libxslt-devel xz-devel \
      snappy-devel \
      postgresql-devel libpng-devel freetype-devel tk-devel \
 && yum clean all \
 && scl enable python27 'virtualenv /opt/dossier-stack'
ENV LD_LIBRARY_PATH /opt/rh/python27/root/usr/lib64

# The default pip in Centos is pretty old, so upgrade it.
RUN /opt/dossier-stack/bin/pip install --upgrade pip \
 && /opt/dossier-stack/bin/pip install --upgrade setuptools

# Install various Python libraries that take a very long time to compile.
RUN /opt/dossier-stack/bin/pip install numpy
RUN /opt/dossier-stack/bin/pip install scipy
RUN /opt/dossier-stack/bin/pip install scikit-learn
RUN /opt/dossier-stack/bin/pip install gevent
RUN /opt/dossier-stack/bin/pip install uwsgi
RUN /opt/dossier-stack/bin/pip install gensim
RUN /opt/dossier-stack/bin/pip install matplotlib
RUN /opt/dossier-stack/bin/pip install nltk

# Download and install the NLTK corpus.
RUN /opt/dossier-stack/bin/python -m nltk.downloader -d /usr/share/nltk_data all

# Now install dossier.models.
RUN /opt/dossier-stack/bin/pip install --pre dossier.models

# Finally, run the dossier.models web server.
EXPOSE 57312
CMD /opt/dossier-stack/bin/uwsgi \
        --http-socket 0.0.0.0:57312 \
        --virtualenv /opt/dossier-stack \
        --wsgi dossier.models.web.wsgi \
        --pyargv "-c /config/config.yaml" \
        --master \
        --processes 10 \
        --vacuum \
        --harakiri 20 \
        --max-requests 5000
