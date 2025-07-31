import pathlib

MODEL_PATH = pathlib.Path(__file__).parents[1] / "models"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
