from __future__ import absolute_import, division, print_function

import argparse
import sys

from dossier.label import LabelStore
from dossier.store import Store
import kvlayer
import yakonfig

from dossier.models.subtopic import Folders


class Factory(yakonfig.factory.AutoFactory):
    config_name = 'sortingdesk_report'
    kvlclient = property(lambda self: kvlayer.client())
    auto_config = lambda self: []


def main():
    p = argparse.ArgumentParser(
        description='SortingDesk report generation tool.')
    p.add_argument('-c', '--config', help='A YAML config file.')
    p.add_argument('-o', '--output', default='-',
                   help='Path to write report file. Defaults to stdout.')
    p.add_argument('folder_name', help='Folder name.')
    p.add_argument('subfolder_name', nargs='?', default=None,
                   help='Subfolder name.')
    args = p.parse_args()

    config = yakonfig.set_default_config([kvlayer], filename=args.config)
    factory = Factory(config)
    store = factory.create(Store)
    label_store = factory.create(LabelStore)
    fout = sys.stdout if args.output == '-' else open(args.output, 'w+')

    subid = None
    if args.subfolder_name is not None:
        subid = Folders.name_to_id(args.subfolder_name)
    generate_report(fout, store, label_store,
                    Folders.name_to_id(args.folder_name), subid)


def generate_report(fout, store, label_store, folder_id, subfolder_id=None):
    '''Writes a report to ``fout``.

    :param fout: A Python file-like object
    :param store: A dossier (feature collection) store.
    :type store: dossier.store.Store
    :param label_store: A label store.
    :type label_store: dossier.label.LabelStore
    '''
    # Use folders API to generate report.
    # The API is documented here: https://dossier-stack.readthedocs.org/en/latest/dossier.web.html#foldering-support-using-dossier-store-and-dossier-label
    # Note though that the `Folders` class in scope here comes from
    # `dossier.models.subtopic`, which subclasses `dossier.web.Folders`.
    # It adds a method for reading subtopic data according to the convention
    # used in SortingDesk. (The implementation of these methods will change
    # once we fix the data race issue.)
    folders = Folders(store, label_store)
    # You probably want to call `folders.subtopics` here.
    return


if __name__ == '__main__':
    main()
