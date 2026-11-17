############################################################################################
def standardInputWrapper(inJson={}):
    """
    Generically unpacks any JSON with the standard params/tables structure.
    Same pattern as Production-Prediction-ML so the UI can send one format to every calculator.
    :param inJson: Raw JSON input dict
    :return:       Flat inputData dict, access any value via inputData['key']
    """
    ########################################################################################
    inputData = {}
    inputData.update({item['name']: item['value']  for item in inJson['data'].get('params', [])})
    inputData.update({item['name']: item['values'] for item in inJson['data'].get('tables', [])})
    ########################################################################################
    return inputData
############################################################################################
