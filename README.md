`dossier.models` is a Python package that provides experimental active learning
models. They are meant to be used as search engines through `dossier.web` web
services.

### Installation

`dossier.models` is on PyPI and can be installed with `pip`:

```bash
pip install dossier.models
```

Currently, `dossier.models` requires Python 2.7. It is not yet Python 3
compatible.


### Documentation

API documentation with examples is available as part of the Dossier Stack
documentation:
[http://dossier-stack.readthedocs.org](http://dossier-stack.readthedocs.org#module-dossier.models)


### Running a simple example

`dossier.models` comes with an example web application that demonstrates how to
use all of the Dossier Stack components to do active learning. The following is
a step-by-step guide to get you up and running with a simple example of
SortingDesk. This guide assumes basic familiarity with standard Python tools
like `pip` and `virtualenv`.

This guide also requires a database of some sort to store data. You can use
any of the backends supported by [kvlayer](https://github.com/diffeo/kvlayer)
(like PostgreSQL, HBase or MySQL). For this guide, we'll use Redis since it
requires very little setup. Just make sure it is installed and running on your
system.

Here are a couple of screenshots of SortingDesk in action:

[![SortingDesk at rest](http://i.imgur.com/I0qT4M9s.png)](http://i.imgur.com/I0qT4M9.png)
[![SortingDesk drag & drop](http://i.imgur.com/Uxeksx5s.png)](http://i.imgur.com/Uxeksx5.png)

First, you should create a new Python virtual environment and install
`dossier.models` from PyPI:

```bash
$ virtualenv dossier
$ source ./dossier/bin/activate
$ pip install dossier.models
```

Depending upon your system setup, this may take a bit of time since
`dossier.models` depends on `numpy`, `scipy` and `scikit-learn`.

Now verify that `dossier.models` is installed correctly:

```bash
$ python -c 'import dossier.models'
```

If all is well, then the command should complete successfully without any
output.

Next, we need to setup configuration so that Dossier Stack knows which database
to use and which indexes to create on feature collections. You can grab
a sample configuration from GitHub:

```bash
$ curl -O https://raw.githubusercontent.com/dossier/dossier.models/master/data/config.yaml
```

The config looks like this:

```yaml
kvlayer:
  app_name: dossier
  namespace: models
  storage_type: redis
  storage_addresses: ['localhost:6379']

dossier.store:
  feature_indexes: ['name', 'keywords']
```

The first section configures your database credentials. This config assumes
you're using Redis running on `localhost` on port `6379` (the default).

The second section tells Dossier Stack which indexes to create on feature
collections. This configuration is dependent on the features in your data.
In this sample configuration, we've chosen `name` and `keywords` because both
are features in the sample data set.

To download and load the sample data set, grab it from GitHub and use the
`dossier.store` command to load it:

```bash
$ curl -O https://raw.githubusercontent.com/dossier/dossier.models/master/data/example.fc
$ dossier.store -c config.yaml load --id-feature content_id example.fc
```

The `dossier.store` command allows you to interact with feature collections
stored in your database. The `--id-feature` flag tells `dossier.store` to use
the value of the `content_id` feature as the feature collection's primary key.
If this flag is omitted, then a `uuid` is generated instead.

You can confirm that data was added to your database with the `ids` command:

```bash
$ dossier.store -c config.yaml ids
doc11
doc12
doc21
doc22
doc23
...
```

Finally, you can run the web application bundled with `dossier.models`:

```bash
$ dossier.models -c config.yaml
```

Open your browser to
[http://localhost:8080/SortingDesk](http://localhost:8080/SortingDesk) to
see an example of `SortingDesk` with the sample data. If you click on the `X`
link on an item in the queue, a negative label will be added between it and the
query indicated at the top of the page. Or you can drag an item from the queue
into a bin---or drop it anywhere on the body page to create a new bin. Bins can
also be dragged on to other bins to merge them. Go ahead and try it. You can
confirm that a label was made with the `dossier.label` command:

```bash
$ dossier.label -c config.yaml list
Label(doc22, doc42, annotator=unknown, 2014-11-26 16:02:01, value=CorefValue.Negative)
```

You should also be able to see labels being added in the output of the
`dossier.models` command if you're running it in your terminal.

There is also a simpler example using plain `SortingQueue` available at
[http://localhost:8080/SortingQueue](http://localhost:8080/SortingQueue).

