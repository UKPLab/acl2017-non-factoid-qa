from experiment.qa.data import QAData
from experiment.qa.data.stackexchange.reader.tsv_reader import TSVReader


class TSVData(QAData):
    def _get_reader(self):
        return TSVReader(self.config['stackexchange'], self.lowercased, self.logger)


component = TSVData
