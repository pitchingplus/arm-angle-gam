import pathlib
from typing import Tuple, Dict, Any

from loguru import logger
import polars as pl
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pygam import LinearGAM, s
import joblib

from armangle.conf import settings
from armangle.predict import predict as predict_arm_angle


def load_and_preprocess_data(training_data: pathlib.Path) -> Tuple[np.ndarray, ...]:
    """
    Load and preprocess the training data

    Args:
        training_data (pathlib.Path): Path to the CSV file containing training data.

    Returns:
        Tuple[np.ndarray, ...]: A tuple containing:
            - features_np (np.ndarray): Features for training and testing.
            - shoulder_x_np (np.ndarray): Shoulder X coordinates.
            - shoulder_z_np (np.ndarray): Shoulder Z coordinates.
            - sample_weights_np (np.ndarray): Sample weights for training data.
            - ball_angle_np (np.ndarray): Ball angles for training and testing.
    """

    df = pl.read_csv(str(training_data))

    features_df = df.select([pl.col("relative_release_ball_x").abs(), "release_ball_z"])
    shoulder_x_df = df.select(["relative_shoulder_x"])
    shoulder_z_df = df.select(["shoulder_z"])
    ball_angle_df = df.select(["ball_angle"])
    sample_weights_series = df.select(["n_pitches"])

    features_np = features_df.to_numpy()
    shoulder_x_np = shoulder_x_df.to_numpy().ravel()
    shoulder_z_np = shoulder_z_df.to_numpy().ravel()
    ball_angle_np = ball_angle_df.to_numpy().ravel()
    sample_weights_np = (
        sample_weights_series.to_numpy() if sample_weights_series is not None else None
    )

    return train_test_split(
        features_np,
        shoulder_x_np,
        shoulder_z_np,
        sample_weights_np,
        ball_angle_np,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )


class OptunaOptimizer:
    def __init__(
        self,
        train_features,
        test_features,
        train_shoulder_x,
        train_shoulder_z,
        train_sample_weights,
        test_ball_angle,
    ):
        self.train_features = train_features
        self.test_features = test_features
        self.train_shoulder_x = train_shoulder_x
        self.train_shoulder_z = train_shoulder_z
        self.train_sample_weights = train_sample_weights
        self.test_ball_angle = test_ball_angle

    def objective(self, trial: optuna.trial.FixedTrial) -> float:
        """Hyperparameter optimization objective

        Args:
            trial (optuna.Trial): The trial object for hyperparameter optimization.

        Returns:
            float: The RMSE of the predicted ball angle.
        """

        gam_shoulder_x = self._create_gam(trial, "x")
        gam_shoulder_z = self._create_gam(trial, "z")

        gam_shoulder_x.fit(
            self.train_features,
            self.train_shoulder_x,
            weights=self.train_sample_weights,
        )
        gam_shoulder_z.fit(
            self.train_features,
            self.train_shoulder_z,
            weights=self.train_sample_weights,
        )

        predicted_shoulder_x = gam_shoulder_x.predict(self.test_features)
        predicted_shoulder_z = gam_shoulder_z.predict(self.test_features)

        predicted_angle_degrees = self._calculate_angle(
            predicted_shoulder_x, predicted_shoulder_z
        )

        return np.sqrt(
            mean_squared_error(self.test_ball_angle, predicted_angle_degrees)
        )

    def _create_gam(self, trial: optuna.trial.FixedTrial, coordinate: str) -> LinearGAM:
        """
        Create a Generalized Additive Model (GAM) for shoulder prediction

        Args:
            trial (optuna.Trial): The trial object for hyperparameter optimization
            coordinate (str): The coordinate to optimize ('x' or 'z')

        Returns:
            LinearGAM: A fitted GAM model for shoulder prediction
        """

        return LinearGAM(
            s(
                0,
                n_splines=trial.suggest_int(
                    f"n_splines_{coordinate}1",
                    settings.N_SPLINES_MIN,
                    settings.N_SPLINES_MAX,
                ),
                lam=trial.suggest_float(
                    f"lam_{coordinate}1", settings.LAM_MIN, settings.LAM_MAX, log=True
                ),
            )
            + s(
                1,
                n_splines=trial.suggest_int(
                    f"n_splines_{coordinate}2",
                    settings.N_SPLINES_MIN,
                    settings.N_SPLINES_MAX,
                ),
                lam=trial.suggest_float(
                    f"lam_{coordinate}2", settings.LAM_MIN, settings.LAM_MAX, log=True
                ),
            )
        )

    def _calculate_angle(self, predicted_shoulder_x, predicted_shoulder_z) -> float:
        """
        Calculate the arm angle based on known release and predicted shoulder coordinates

        Args:
            predicted_shoulder_x (np.ndarray): Predicted shoulder X coordinates
            predicted_shoulder_z (np.ndarray): Predicted shoulder Z coordinates

        Returns:
            float: The predicted arm angle in degrees
        """

        release_ball_x_test = self.test_features[:, 0]
        release_ball_z_test = self.test_features[:, 1]

        predicted_angle_radians = np.arctan(
            (release_ball_z_test - predicted_shoulder_z)
            / (release_ball_x_test - predicted_shoulder_x + settings.EPSILON)
        )

        return np.degrees(predicted_angle_radians)

    def optimize(self) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna

        Returns:
            dict: The best hyperparameters found during optimization
        """

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=settings.N_TRIALS, n_jobs=-1)

        return study.best_params


def create_final_models(
    best_params: Dict[str, Any],
    train_features,
    train_shoulder_x,
    train_shoulder_z,
    train_sample_weights,
) -> tuple[LinearGAM, LinearGAM]:
    """
    Create and train final models using the best parameters.

    Args:
        best_params (dict): Best hyperparameters from the optimization.
        train_features (np.ndarray): Training features.
        train_shoulder_x (np.ndarray): Training shoulder X coordinates.
        train_shoulder_z (np.ndarray): Training shoulder Z coordinates.
        train_sample_weights (np.ndarray): Sample weights for training data.

    Returns:
        tuple: Final models for shoulder X and Z coordinates.
    """

    final_gam_shoulder_x = LinearGAM(
        s(0, n_splines=best_params["n_splines_x1"], lam=best_params["lam_x1"])
        + s(1, n_splines=best_params["n_splines_x2"], lam=best_params["lam_x2"])
    )
    final_gam_shoulder_x.fit(
        train_features, train_shoulder_x, weights=train_sample_weights
    )

    final_gam_shoulder_z = LinearGAM(
        s(0, n_splines=best_params["n_splines_z1"], lam=best_params["lam_z1"])
        + s(1, n_splines=best_params["n_splines_z2"], lam=best_params["lam_z2"])
    )
    final_gam_shoulder_z.fit(
        train_features, train_shoulder_z, weights=train_sample_weights
    )

    return final_gam_shoulder_x, final_gam_shoulder_z


def calculate_metrics(test_data, predictions):
    """
    Calculate and return RMSE and R² for given test data and predictions.

    Args:
        test_data (np.ndarray): The actual test data.
        predictions (np.ndarray): The predicted values.

    Returns:
        tuple: RMSE and R² scores.
    """

    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    r2 = r2_score(test_data, predictions)

    return rmse, r2


def train(training_data: pathlib.Path):
    """
    Train the ball angle predictor model using the provided training data.

    Args:
        training_data (pathlib.Path): Path to the CSV file containing training data.
    """

    (
        train_features,
        test_features,
        train_shoulder_x,
        test_shoulder_x,
        train_shoulder_z,
        test_shoulder_z,
        train_sample_weights,
        test_sample_weights,
        train_ball_angle,
        test_ball_angle,
    ) = load_and_preprocess_data(training_data)

    optimizer = OptunaOptimizer(
        train_features,
        test_features,
        train_shoulder_x,
        train_shoulder_z,
        train_sample_weights,
        test_ball_angle,
    )
    best_params = optimizer.optimize()

    logger.debug(
        "Best RMSE for ball angle prediction: {:.4f}".format(
            optimizer.objective(optuna.trial.FixedTrial(best_params))
        )
    )
    logger.debug("Best hyperparameters: {}".format(best_params))

    logger.info("Creating models with best hyperparameters")
    final_gam_shoulder_x, final_gam_shoulder_z = create_final_models(
        best_params,
        train_features,
        train_shoulder_x,
        train_shoulder_z,
        train_sample_weights,
    )

    final_predicted_shoulder_x = final_gam_shoulder_x.predict(test_features)
    final_predicted_shoulder_z = final_gam_shoulder_z.predict(test_features)

    joblib.dump(final_gam_shoulder_x, settings.SHOULDER_X_MODEL_PATH)
    logger.debug("Saved final_gam_shoulder_x model: {}", settings.SHOULDER_X_MODEL_PATH)

    joblib.dump(final_gam_shoulder_z, settings.SHOULDER_Z_MODEL_PATH)
    logger.debug("Saved final_gam_shoulder_z model: {}", settings.SHOULDER_Z_MODEL_PATH)

    logger.debug("Running quality check")
    final_predicted_angle_degrees = predict_arm_angle(
        final_gam_shoulder_x, final_gam_shoulder_z, test_features
    )

    rmse_shoulder_x, r2_shoulder_x = calculate_metrics(
        test_shoulder_x, final_predicted_shoulder_x
    )
    rmse_shoulder_z, r2_shoulder_z = calculate_metrics(
        test_shoulder_z, final_predicted_shoulder_z
    )
    rmse_ball_angle, r2_ball_angle = calculate_metrics(
        test_ball_angle, final_predicted_angle_degrees
    )

    quality_banner = f"""
    Performance Metrics for Shoulder Predictions:

        Relative Shoulder X:
          RMSE = {rmse_shoulder_x:.4f}
          R² = {r2_shoulder_x:.4f}
          
        Shoulder Z:
          RMSE = {rmse_shoulder_z:.4f}
          R² = {r2_shoulder_z:.4f}
      
    Performance Metrics for Final Ball Angle Prediction:

        Ball Angle:
          RMSE = {rmse_ball_angle:.4f}
          R² = {r2_ball_angle:.4f}
    """.strip()

    logger.info(quality_banner)


if __name__ == "__main__":
    training_data_path = pathlib.Path("~/src/compute-arm-angle/savant-angles-all.csv")
    train(training_data_path)
