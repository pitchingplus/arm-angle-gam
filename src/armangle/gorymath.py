import numpy as np

from armangle.conf import settings


def calculate_arm_angle(
    release_ball_x: float,
    release_ball_z: float,
    predicted_shoulder_x: float,
    predicted_shoulder_z: float,
    epsilon: float = settings.EPSILON,
) -> float:
    """
    Arm angle calculation formula

    Args:
        release_ball_x: x release coordinate of the ball
        release_ball_z: z release coordinate of the ball
        predicted_shoulder_x: x-coordinate of the predicted shoulder location
        predicted_shoulder_z: z-coordinate of the predicted shoulder location
        epsilon: A small value to avoid division by zero

    Returns:
        The angle in degrees between the line from the shoulder to the ball and
        the horizontal plane.
    """

    predicted_angle_radians = np.arctan(
        (release_ball_z - predicted_shoulder_z)
        / (release_ball_x - predicted_shoulder_x + epsilon)
    )

    return np.degrees(predicted_angle_radians)
