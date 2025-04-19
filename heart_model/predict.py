import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from heart_model import __version__ as _version
from heart_model.config.core import config
from heart_model.processing.data_manager import load_pipeline
from heart_model.processing.data_manager import pre_pipeline_preparation
from heart_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
heart_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = heart_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    data_in = [{
                    "age": 75,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 157,
                    "chol": 131,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 141,
                    "exang": 1,
                    "oldpeak": 3.4,
                    "slope": 2,
                    "ca": 2,
                    "thal": "fixed",
                }]
    # 57,1,4,130,131,0,0,115,1,1.2,2,1,reversible,1
    # 68,1,4,144,193,1,0,141,0,3.4,2,2,reversible,1

    make_prediction(input_data = data_in)