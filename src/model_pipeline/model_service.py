from pathlib  import Path
import pickle as pkl
from model_pipeline.model_train import ModelTrainer
from loguru import logger
from model_pipeline.collection import load_data_from_csv
from settings import settings


class ModelService:
    def __init__(self, df):
        """
        df: Raw DataFrame to train model if it doesn't exist
        """
        self.df = df
        self.model = None

    def load_model(self):
        model_path = Path(f"{settings.model_path}/{settings.model_name}")

        logger.info(f"Checking for model file at: {model_path}")

        if not model_path.exists():
            logger.warning(f"Model not found — training new model: {settings.model_name}")
            trainer = ModelTrainer(self.df)
            trainer.train(run_name="RandomForest_lagged")
            trainer.save_model(model_path)  # save after training

        logger.info(f"Model found! Loading from: {model_path}")
        with open(model_path, "rb") as f:
            self.model = pkl.load(f)

    def predict(self, input_parameters):
        """
        input_parameters: list or array of input features
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before calling predict()")

        logger.info("Making prediction...")
        return self.model.predict([input_parameters])