import pathlib
from typing import Union

import joblib
import numpy as np
from pygam import LinearGAM

from armangle.conf import settings
from armangle.gorymath import calculate_arm_angle


TModelOrPath = Union[LinearGAM, pathlib.Path]


def predict(
    shoulder_x_model: TModelOrPath,
    shoulder_z_model: TModelOrPath,
    release_features: np.array,
    epsilon: float = settings.EPSILON,
) -> np.array:
    """
    Predicts the ball angle given the release features.

    Args:
        shoulder_x_model: A fitted model for shoulder x-coordinate prediction.
            LinearGAM or a path to a saved model.
        shoulder_z_model: A fitted model for shoulder z-coordinate prediction.
            LinearGAM or a path to a saved model.
        release_features: A NumPy array with two columns:
            - 0: relative_release_ball_x
            - 1: release_ball_z
        epsilon: A small value to avoid division by zero.

    Returns:
        A NumPy array of predicted ball angles in degrees.
    """

    if isinstance(shoulder_x_model, pathlib.Path):
        shoulder_x_model = joblib.load(shoulder_x_model)
    if isinstance(shoulder_z_model, pathlib.Path):
        shoulder_z_model = joblib.load(shoulder_z_model)

    # reflect release point x coordinate
    release_features[:, 0] = np.abs(release_features[:, 0])

    predicted_shoulder_x = shoulder_x_model.predict(release_features)
    predicted_shoulder_z = shoulder_z_model.predict(release_features)

    relative_release_ball_x = release_features[:, 0]
    release_ball_z = release_features[:, 1]

    return calculate_arm_angle(
        release_ball_x=relative_release_ball_x,
        release_ball_z=release_ball_z,
        predicted_shoulder_x=predicted_shoulder_x,
        predicted_shoulder_z=predicted_shoulder_z,
        epsilon=epsilon,
    )
