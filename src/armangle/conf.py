import pathlib
from typing import Annotated

from pydantic import ConfigDict, BeforeValidator
from pydantic_settings import BaseSettings


def ensure_path_exists(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="ARM_ANGLE_")

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_TRIALS: int = 100
    EPSILON: float = 1e-8

    # Hyperparameter search spaces
    N_SPLINES_MIN: int = 5
    N_SPLINES_MAX: int = 20
    LAM_MIN: float = 1e-3
    LAM_MAX: float = 1e2

    SHOULDER_X_MODEL_PATH: Annotated[
        pathlib.Path, BeforeValidator(ensure_path_exists)
    ] = pathlib.Path("models/shoulder_x_model.joblib")
    SHOULDER_Z_MODEL_PATH: Annotated[
        pathlib.Path, BeforeValidator(ensure_path_exists)
    ] = pathlib.Path("models/shoulder_z_model.joblib")


settings = Settings()  # noqa
