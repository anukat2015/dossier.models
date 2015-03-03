#!/usr/bin/python
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
        description='SortingDesk report generation tool')
    p.add_argument('-c', '--config', required=True,
                   help='Dossier stack YAML config file')
    p.add_argument('folder_name', nargs='?', default=None, help='Folder name')
    p.add_argument('-u', '--user', default=None,help='user name (default=ALL)')
    args = p.parse_args()

    config = yakonfig.set_default_config([kvlayer], filename=args.config)
    factory = Factory(config)
    store = factory.create(Store)
    label_store = factory.create(LabelStore)
    folders = Folders(store, label_store)

    list_projects(folders, args.folder_name, args.user)


def list_projects(folders, folder = None, user = None):
    '''List all folders or all subfolders of a folder.

    If folder is provided, this method will output a list of subfolders
    contained by it.  Otherwise, a list of all top-level folders is produced.

    :param folders: reference to folder.Folders instance
    :param folder: folder name or None
    :param user: optional user name

    '''
    fid = None if folder is None else Folders.name_to_id(folder)

    # List all folders if none provided.
    if fid is None:
        for f in folders.folders(user):
            print(Folders.id_to_name(f))

        return

    # List subfolders of a specific folder
    try:
        for sid in folders.subfolders(fid, user):
            print(Folders.id_to_name(sid))
    except KeyError:
        print("E: folder not found: %s" %folder, file=sys.stderr)


if __name__ == '__main__':
    main()
