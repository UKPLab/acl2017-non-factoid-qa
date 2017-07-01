from experiment.qa.data import QAData
from experiment.qa.data.insuranceqa.reader.v2_reader import V2Reader


class V2Data(QAData):
    def _get_reader(self):
        return V2Reader(self.config['insuranceqa'], self.lowercased, self.logger)


component = V2Data
