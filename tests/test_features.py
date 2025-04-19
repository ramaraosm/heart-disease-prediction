
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from heart_model.config.core import config
from heart_model.processing.features import Mapper


def test_thal_variable_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.model_config_.thal_var, 
                    mappings = config.model_config_.thal_mappings)
    assert sample_input_data[0].loc[70, 'thal'] == 'reversible'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[70, 'thal'] == 2