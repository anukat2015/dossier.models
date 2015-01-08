import os

from setuptools import setup, find_packages

from version import get_git_version
VERSION, SOURCE_LABEL = get_git_version()
PROJECT = 'dossier.models'
AUTHOR = 'Diffeo, Inc.'
AUTHOR_EMAIL = 'support@diffeo.com'
URL = 'http://github.com/dossier/dossier.models'
DESC = 'Active learning models'


def read_file(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    return open(file_path).read()


setup(
    name=PROJECT,
    version=VERSION,
    description=DESC,
    license='MIT',
    long_description=read_file('README.md'),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License',
    ],
    install_requires=[
        'beautifulsoup4',
        'dossier.fc >= 0.1.4',
        'dossier.label >= 0.1.5',
        'dossier.web >= 0.5.0',
        'happybase',
        'joblib',
        'nltk',
        'numpy',
        'regex',
        'requests',
        'scipy',
        'scikit-learn',
        'streamcorpus-pipeline',
        'pytest',
        'pytest-diffeo >= 0.1.4',
    ],
    extras_require={
        'tfidf': ['gensim'],
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'dossier.models = dossier.models.web.__main__:main',
            'dossier.etl = dossier.models.etl:main',
        ],
    },
)
