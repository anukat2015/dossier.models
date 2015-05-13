#!/usr/bin/python
from __future__ import absolute_import, division, print_function

import argparse
from io import BytesIO
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import sys
from urllib2 import urlopen
import re

from PIL import Image
import xlsxwriter

from dossier.store import Store
from dossier.models.folder import Folders
from dossier.models.subtopic import subtopics
import kvlayer
import yakonfig


class Factory(yakonfig.factory.AutoFactory):
    config_name = 'sortingdesk_report'
    kvlclient = property(lambda self: kvlayer.client())
    auto_config = lambda self: []


def main():
    p = argparse.ArgumentParser(
        description='SortingDesk report generation tool')
    p.add_argument('-c', '--config', required=True,
                   help='dossier stack YAML config file')
    p.add_argument('-o', '--output', required=True,
                   help='path to write Excel workbook file')
    p.add_argument(
        '-u', '--user', default='unknown', help='user name (default=ALL)')
    p.add_argument('folder', help='folder name')
    p.add_argument('subfolder', nargs='?', default=None,
                   help='subfolder name (default=ALL)')
    args = p.parse_args()

    config = yakonfig.set_default_config([kvlayer], filename=args.config)
    factory = Factory(config)
    store = factory.create(Store)

    # Instantiate and run report generator.
    folders = Folders(kvlayer.client())
    gen = ReportGenerator(store, folders, args.folder,
                          subfolder_name=args.subfolder, user=args.user)
    with open(args.output, 'wb+') as out:
        gen.run(out)


class ReportGenerator(object):
    '''Generates a report in Excel format.'''

    def __init__(self, store, folders, folder_name, subfolder_name=None, user='unknown'):
        '''Class constructor.

        :param folders: Reference to folder.Folders instance
        :param folder: folder name to generate report for
        :param subfolder: subfolder name; must be contained by folder and can
                          be None
        :param user: Generate report on data created by specified user.
        '''
        self.store = store
        self.folders = folders
        self.workbook = None
        self.formats = {}
        self.folder_name = folder_name
        self.subfolder_name = subfolder_name
        self.user = user

    @property
    def folder_id(self):
        return Folders.name_to_id(self.folder_name)

    @property
    def subfolder_id(self):
        return Folders.name_to_id(self.subfolder_name)

    def run(self, output):
        '''Generate the report to the given output.

        :param output: writable file-like object or file path
        '''
        # Ensure folder exists.
        if self.folder_id not in self.folders.folders(self.user):
            print("E: folder not found: %s" % self.folder_name,
                  file=sys.stderr)
            return

        # Create workbook.
        wb = self.workbook = xlsxwriter.Workbook(output)

        # Create the different styles used by this report generator.
        self.formats['title'] = wb.add_format({'font_size': '18',
                                               'bold': True})

        self.formats['default'] = wb.add_format({'align': 'top'})
        self.formats['bold'] = wb.add_format({'bold': True})

        self.formats['header'] = wb.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'top',
            'font_size': '14',
            'font_color': '#506050',
            'bg_color': '#f5f5f5',
            'right': 1,
            'border_color': 'white'})

        self.formats['pre'] = wb.add_format({'font_name': 'Courier',
                                             'valign': 'top'})

        self.formats['link'] = wb.add_format({'valign': 'top',
                                              'font_color': 'blue',
                                              'underline': True})

        self.formats['type_text'] = wb.add_format({
            'font_color': '#BF8645',
            'valign': 'top',
            'align': 'center'})

        self.formats['type_image'] = wb.add_format({
            'font_color': '#84BF45',
            'valign': 'top',
            'align': 'center'})

        # Generate report for a specific subfolder or *all* subfolders of
        # self.folder .
        if self.subfolder_id is None:
            self._generate_report_all()
        else:
            self._generate_report_single(self.subfolder_id)

        # done and outta here
        self.workbook.close()

    def _generate_report_all(self):
        '''Generate report for all subfolders contained by self.folder_id.'''
        assert self.workbook is not None
        count = 0

        # Do all subfolders
        for sid in self.folders.subfolders(self.folder_id, self.user):
            count += 1
            self._generate_for_subfolder(sid)

        if count == 0:
            print("I: empty workbook created: no subfolders found")

    def _generate_report_single(self, sid):
        '''Generate report for subfolder given by sid .

        The main purpose of this method is to make sure the subfolder given by
        sid does indeed exist.  All real work is delegated to
        _generate_for_subfolder.

        :param sid: The subfolder id

        Private method.
        '''
        assert self.workbook is not None
        assert sid is not None

        # Ensure subfolder exists
        if not sid in self.folders.subfolders(self.folder_id, self.user):
            subfolder = Folders.id_to_name(sid)
            print("E: subfolder not found: %s" % subfolder, file=sys.stderr)
            return

        self._generate_for_subfolder(sid)

    def _generate_for_subfolder(self, sid):
        ''' Generate report for a subfolder.

        :param sid: The subfolder id; assumed valid
        '''
        # TODO: the following assumes subfolder names can be constructed from a
        # subfolder id, which might not be the case in the future.
        name = self._sanitise_sheetname(uni(Folders.id_to_name(sid)))
        ws = self.workbook.add_worksheet(name)
        fmt = self.formats
        ws.write("A1", "Dossier report", fmt['title'])
        ws.write("A2", "%s | %s" % (uni(self.folder_name), name))

        # Column dimensions
        ws.set_column('A:A', 37)
        ws.set_column('B:B', 37)
        ws.set_column('C:C', 37)
        ws.set_column('D:D', 8)
        ws.set_column('E:E', 30)
        ws.set_column('F:F', 37)

        # Header
        ws.write("A4", "Id", fmt['header'])
        ws.write("B4", "URL", fmt['header'])
        ws.write("C4", "Subtopic Id", fmt['header'])
        ws.write("D4", "Type", fmt['header'])
        ws.write("E4", "Content", fmt['header'])
        ws.write("F4", "Image URL", fmt['header'])

        # TODO: we probably want to wrap the following in a try-catch block, in
        # case the call to folders.subtopics fails.
        row = 4
        for i in subtopics(self.store, self.folders, self.folder_id, sid, self.user):
            Item.construct(self, i).generate_to(ws, row)
            row += 1

    def _sanitise_sheetname(self, sheetname):
        '''Sanitize worksheet names.

        The length of the sheetname is kept within 31 characters and any
        invalid chars are replaced by underscores.

        '''
        # Ensure that invalid characters are converted to underscore, whilst
        # making sure sheetname's length falls within 31 chars.
        return re.sub(r"[\[\]:*?/\\]", "_", sheetname[:31])


class Item(object):
    ''' Base class of concrete items ItemText and ItemImage. '''

    @staticmethod
    def construct(generator, subtopic):
        '''Method constructor of Item-derived classes.

        Given a subtopic tuple, this method attempts to construct an
        Item-derived class, currently either ItemText or ItemImage, from the
        subtopic's type, found in its 4th element.

        :param generator: Reference to the owning ReportGenerator instance
        :param subtopic: Tuple containing content_id, meta_url, subtopic_id,
        type and type-specific data.

        :returns An instantiated Item-derived class.

        '''
        type = subtopic[3]
        if type not in Item.constructors:
            raise LookupError(type)   # perhaps customise this exception?

        return Item.constructors[type](generator, subtopic)

    def __init__(self, generator, subtopic):
        self.generator = generator
        self.content_id, self.subtopic_id, self.meta_url, \
            type, self.data = subtopic[0:5]

    def generate_to(self, worksheet, row):
        fmt = self.generator.formats
        worksheet.write(row, 0, uni(self.content_id), fmt['pre'])
        worksheet.write(row, 1, uni(self.meta_url), fmt['link'])
        worksheet.write(row, 2, uni(self.subtopic_id), fmt['pre'])


class ItemImage(Item):
    ''' Represents an image item for the purpose of report generation. '''

    def __init__(self, generator, subtopic):
        '''Constructor.

        Delegates to base class constructor.

        '''
        super(ItemImage, self).__init__(generator, subtopic)

    def generate_to(self, worksheet, row):
        '''Generate row report.

        Generates a row report of the item represented by this instance and
        inserts it into a given worksheet at a specified row number.

        :param worksheet: Reference to a worksheet in which to insert row
        report.

        :param row: Row number.
        '''
        super(ItemImage, self).generate_to(worksheet, row)

        embedded = False
        fmt = self.generator.formats
        worksheet.write(row, 3, "image", fmt['type_image'])
        worksheet.write_url(row, 5, self.data[0], fmt['link'])

        try:
            if self.data:
                image = self.resize_image(StringIO(self.data[1]))
                worksheet.insert_image(row, 4, 'image',
                                       {'image_data': image})
                embedded = True
        except:
            # We probably wrongly ignoring the exception.  Should really at
            # least log it somewhere.
            pass

        # Attempt to retrieve image data from the image URL if image data not
        # present already above or something failed whilst recreating image
        # from base64 encoding.
        if not embedded:
            url = self.data[0]
            if url:
                image = self.resize_image(BytesIO(urlopen(url).read()))
                worksheet.insert_image(row, 4, url, {'image_data': image})
                embedded = True
            else:
                worksheet.write(row, 4, '<unavailable>')

        if embedded:
            worksheet.set_row(row, 40)

    def resize_image(self, data):
        '''Resize image if height over 50 pixels and convert to JPEG.

        Given a ByteIO or StringIO data input, this method ensures that the
        image is not over 50 pixels high.  If it is over 50 pixels high, the
        image is resized to precisely 50 pixels in height and the width is
        adjusted accordingly in keeping with the width/height ratio.  The image
        is always converted to JPEG to minimize any potentials issues while
        embedding the image in the Excel workbook.

        :param data: ByteIO or StringIO stream containing image data
        :returns Reference to a BytesIO instance containing resized image data.

        '''
        image = Image.open(data)
        stream_out = BytesIO()
        width, height = image.size[:]
        if height > 50:
            width = int(width * 50 / height)
            height = 50
            image = image.resize((width, 50))

        image.save(stream_out, format="JPEG", quality=100)

        stream_out.seek(0)
        return stream_out


class ItemText(Item):
    '''Represents a text snippet item for the purpose of report generation.'''

    def __init__(self, generator, subtopic):
        '''Constructor.

        Delegates to base class constructor.
        '''
        super(ItemText, self).__init__(generator, subtopic)

    def generate_to(self, worksheet, row):
        '''Generate row report.

        Generates a row report of the item represented by this instance and
        inserts it into a given worksheet at a specified row number.

        :param worksheet: Reference to a worksheet in which to insert row
        report.

        :param row: Row number.

        '''
        super(ItemText, self).generate_to(worksheet, row)

        fmt = self.generator.formats
        worksheet.write(row, 3, "text", fmt['type_text'])
        worksheet.write(row, 4, uni(self.data), fmt['default'])


class ItemManual(ItemText):
    pass


def uni(t):
    if t is None:
        return None
    if isinstance(t, unicode):
        return t
    return unicode(t, 'utf-8')


Item.constructors = {
    "text": ItemText,
    "image": ItemImage,
    "manual": ItemManual,
}
'''Map (dict) holding references to class constructors per item type.'''


if __name__ == '__main__':
    main()
