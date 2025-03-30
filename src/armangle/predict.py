import numpy as np
from pygam import LinearGAM

from armangle.conf import settings


def predict(
    shoulder_x_model: LinearGAM,
    shoulder_z_model: LinearGAM,
    release_features: np.array,
    epsilon: float = settings.EPSILON,
) -> np.array:
    """
    Predicts the ball angle given the release features.

    Args:
        shoulder_x_model: A fitted model for shoulder x-coordinate prediction.
        shoulder_z_model: A fitted model for shoulder z-coordinate prediction.
        release_features: A NumPy array with two columns:
            - Column 0: relative_release_ball_x
            - Column 1: release_ball_z
        epsilon: A small value to avoid division by zero.

    Returns:
        A NumPy array of predicted ball angles in degrees.
    """

    # reflect release point x coordinate
    release_features[:, 0] = np.abs(release_features[:, 0])

    predicted_shoulder_x = shoulder_x_model.predict(release_features)
    predicted_shoulder_z = shoulder_z_model.predict(release_features)

    relative_release_ball_x = release_features[:, 0]
    release_ball_z = release_features[:, 1]

    predicted_angle_radians = np.arctan(
        (release_ball_z - predicted_shoulder_z)
        / (relative_release_ball_x - predicted_shoulder_x + epsilon)
    )
    predicted_angle_degrees = np.degrees(predicted_angle_radians)

    return predicted_angle_degrees
