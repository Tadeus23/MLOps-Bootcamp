import pandas as pd
import numpy as np
import joblib
from config import config
from processing.data_handling import load_pipeline, load_dataset

classification_pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)

def generate_predictions(data_input):
    data = pd.DateFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1, 'Y', 'N')
    result = {"prediction": output}
    return result


# def generate_predictions():
#     test_data = load_dataset(config.TEST_FILE)
#     pred = classification_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(pred==1, 'Y', 'N')
#     print(output)
#     # result = {"Predictions": output}
#     return output

if __name__=='main':
    generate_predictions()