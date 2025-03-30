# Ball Angle Predictor

This project implements a machine learning model to predict the ball angle in baseball pitches using Generalized Additive Models (GAMs) and hyperparameter optimization with Optuna.

## Workflow

1. **Data Loading and Preprocessing**
   - The training data is loaded from a CSV file.
   - Features (`relative_release_ball_x` and `release_ball_z`) and targets (`shoulder_x`, `shoulder_z`, and `ball_angle`) are extracted.
   - Data is split into training and testing sets.

2. **Hyperparameter Optimization**
   - Optuna is used to find the best hyperparameters for the GAM models.
   - Two separate GAMs are optimized: one for predicting `shoulder_x` and another for `shoulder_z`.
   - The objective function calculates the RMSE of the predicted ball angle.

3. **Model Training**
   - Final GAM models for `shoulder_x` and `shoulder_z` are created using the best hyperparameters.
   - Models are trained on the full training dataset.

4. **Model Evaluation**
   - The trained models are used to make predictions on the test set.
   - RMSE and R² scores are calculated for `shoulder_x`, `shoulder_z`, and `ball_angle` predictions.

5. **Model Saving**
   - The final trained models are saved using `joblib` for later use.

## Usage

### Training

Grab a CSV of arm angle data from Savant: https://baseballsavant.mlb.com/leaderboard/pitcher-arm-angles?batSide=&dateStart=&dateEnd=&gameType=R%7CF%7CD%7CL%7CW&groupBy=year%7Capi_pitch_type_group03&min=q&minGroupPitches=1&perspective=back&pitchHand=&pitchType=&season=2024%7C2023%7C2022%7C2021%7C2020&size=small&sort=ascending&team=. This is our training data.

To train the model:

```python
from pathlib import Path
from armangle.train import train

training_data_path = Path("path/to/savant-arm-angles.csv")
train(training_data_path)
```

This will:
1. Load and preprocess the data
2. Perform hyperparameter optimization
3. Train the final models
4. Print performance metrics
5. Save the trained models to paths defined in the settings (see below)

### Running predictions

To run predictions using the trained models, you can use the `predict` function from the `armangle.predict` module. Make sure to load the models first.

```python
import numpy as np
import joblib
from armangle.conf import settings
from armangle.predict import predict


shoulder_x_model = joblib.load(settings.SHOULDER_X_MODEL_PATH)
shoulder_z_model = joblib.load(settings.SHOULDER_Z_MODEL_PATH)

release_point = np.array([
  [-2.0167545160115523, 5.683197115],
])

predicted_arm_angle = predict(
    shoulder_x_model,
    shoulder_z_model,
    release_point,
)

# expected arm angle: 30.7
# predicted_arm_angle: np.array([32.78066166])
```

## Model Components

1**GAM for shoulder_x**: Predicts the relative shoulder X coordinate. Saved and loaded using `joblib` from `SHOULDER_X_MODEL_PATH` setting.
2**GAM for shoulder_z**: Predicts the shoulder Z coordinate. Saved and loaded using `joblib` from `SHOULDER_Z_MODEL_PATH` setting.

## Configuration

The `settings` object from `armangle.conf` is used to configure various parameters such as:
- EPSILON: Used to avoid division by zero in angle calculations
- TEST_SIZE: Proportion of data to use for testing
- RANDOM_STATE: Seed for random number generation
- N_SPLINES_MIN and N_SPLINES_MAX: Range for number of splines in GAM models
- LAM_MIN and LAM_MAX: Range for lambda (regularization) parameter in GAM models
- N_TRIALS: Number of trials for Optuna optimization
- SHOULDER_X_MODEL_PATH: Path to save the shoulder-x model
- SHOULDER_Z_MODEL_PATH: Path to save the shoulder-z model

Adjust these settings in the `conf.py` file to fine-tune the training process.
