import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from heart_model.config.core import config
from heart_model.processing.features import Mapper


heart_pipe = Pipeline([
    
    ######### Mapper ###########
    ('map_thal', Mapper(variable = config.model_config_.thal_var, mappings = config.model_config_.thal_mappings)),
   

    # Scale features
    ('scaler', StandardScaler()),
    
    # Regressor
    ('model_rf', RandomForestRegressor(n_estimators = config.model_config_.n_estimators, 
                                       max_depth = config.model_config_.max_depth,
                                      random_state = config.model_config_.random_state))
    
    ])
