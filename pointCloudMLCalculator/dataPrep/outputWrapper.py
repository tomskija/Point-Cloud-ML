############################################################################################
import json
import os
############################################################################################
class standardOutputWrapper:
    ########################################################################################
    def __init__(self):
        self.params = []
        self.tables = []
    ########################################################################################
    def add_param(self, name='', value=None, value_type=2):
        self.params.append({'name': name, 'value': value, 'value_type': value_type})
    ########################################################################################
    def add_table(self, name='', array=[], row_headers=[], column_headers=[], shape=[]):
        self.tables.append({'name': name, 'array': array, 'row_headers': row_headers, 'column_headers': column_headers, 'shape': shape if shape else [len(array)]})
    ########################################################################################
    def flatten(self):
        """Params and tables in one dict, used for test assertions."""
        result = {p['name']: p['value'] for p in self.params}
        result.update({t['name']: t['array'] for t in self.tables})
        return result
    ########################################################################################
    def dump_json(self, fileName=''):
        """Serialize output_wrapper to JSON file, used for test fixture generation."""
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        with open(fileName, 'w') as f:
            json.dump({'params': self.params, 'tables': self.tables}, f, indent=4)
############################################################################################
