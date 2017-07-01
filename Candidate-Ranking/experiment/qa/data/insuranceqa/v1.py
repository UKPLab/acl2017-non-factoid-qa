from experiment.qa.data import QAData
from experiment.qa.data.insuranceqa.reader.v1_reader import V1Reader


class V1Data(QAData):
    def _get_reader(self):
        return V1Reader(self.config['insuranceqa'], self.lowercased, self.logger)


component = V1Data
