# An unbelievably disgusting compatibility layer because I'm being rushed.
# ---AG

from __future__ import absolute_import, division, print_function

from collections import defaultdict

import dossier.web as web
import yakonfig


class Folders(web.Folders):
    DEFAULT_ANNOTATOR_ID = 'unknown'

    def __init__(self, *args, **kwargs):
        # A horrible hack to create a new `Folders` instance with config.
        try:
            config = yakonfig.get_global_config('dossier.folders')
            # For old configs.
            if 'prefix' in config:
                config['namespace'] = config.pop('prefix')
        except KeyError:
            config = {}
        super(Folders, self).__init__(*args, **dict(config, **kwargs))

    @staticmethod
    def name_to_id(v):
        return v

    @staticmethod
    def id_to_name(v):
        return v

    def folders(self, *args, **kwargs):
        return [it.name for it in self.list('/') if it.is_folder()]

    def subfolders(self, fid, *args, **kwargs):
        return [it.name for it in self.list(fid) if it.is_folder()]

    def items(self, fid, subid, *args, **kwargs):
        vals = []
        for it in self.list(fid + '/' + subid):
            if '@' in it.name:
                vals.append(tuple(map(lambda s: s.encode('utf-8'),
                                      it.name.split('@'))))
            else:
                vals.append((it.name.encode('utf-8'), None))
        return vals

    def grouped_items(self, fid, subid, *args, **kwargs):
        d = defaultdict(list)
        for cid, subid in self.items(fid, subid):
            d[cid].append(subid)
        return d

    def parent_subfolders(self, ident, *args, **kwargs):
        cid, _ = normalize_ident(ident)

        # There's no index structure to find parents of one path component.
        # So we need to do an exhaustive search. Yay!
        vals = []
        for folder in list(self.list('/')):
            if not folder.is_folder():
                continue
            for subfolder in list(self.list(folder.path)):
                if not subfolder.is_folder():
                    continue
                for item in list(self.list(subfolder.path)):
                    if not item.is_item():
                        continue
                    if cid in item.name.split('@'):
                        vals.append((folder.name, subfolder.name))
        return vals


def normalize_ident(ident):
    '''Splits a generic identifier.

    If ``ident`` is a tuple, then ``(ident[0], ident[1])`` is returned.
    Otherwise, ``(ident[0], None)`` is returned.
    '''
    if isinstance(ident, tuple) and len(ident) == 2:
        return ident[0], ident[1]  # content_id, subtopic_id
    else:
        return ident, None
