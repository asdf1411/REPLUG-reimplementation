# encoding = utf-8
import json
from tqdm import tqdm

class Reader(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.top_k = 5
        # stream_data: a list of string
        self.raw_data = self._get_raw_data()
        self.stream_data = self._get_stream_data()

    def _get_raw_data(self):
        with open(self.data_path, 'r') as fr:
            filter_instance = json.load(fr)
        fr.close()

        return filter_instance

    def _get_stream_data(self):
        queries = []
        for index, item in tqdm(enumerate(self.raw_data)):
            query = self.raw_data[item]['question']
            queries.append(query)

        return queries