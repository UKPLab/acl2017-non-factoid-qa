import gzip


class ArchiveReader(object):
    def __init__(self, archive_path, lowercased, logger):
        self.archive_path = archive_path
        self.lowercased = lowercased
        self.logger = logger

    def read(self):
        raise NotImplementedError()


class TSVArchiveReader(ArchiveReader):
    def read_tsv(self, file_path, is_gzip=False):
        result = []

        if is_gzip:
            f = gzip.open(file_path, 'r')
        else:
            f = open(file_path, 'r')

        try:
            for line in f:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                line = line.rstrip()
                if self.lowercased:
                    line = line.lower()
                if line:
                    result.append(line.split('\t'))
        finally:
            f.close()

        return result
