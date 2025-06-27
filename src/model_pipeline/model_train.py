import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from loguru import logger
import pickle as pkl
from model_pipeline.preparation import DataPrepper
import mlflow
import mlflow.sklearn


class ModelTrainer:

    """
    Class to train and evaluate a regression model using the DataPrepper pipeline
    """

    def __init__(self, df,target_col='active_power',
              model=None, test_size=0.2,random_state=42):
    
    
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

    def preprocess(self):
        """
        utilize DataPrepper class to prepare data for non Pipeline preprocessing
        """
        logger.info("Preprocessing data using DataPrepper...")
        prepper = DataPrepper(self.df)
        df_prepared = prepper.transform()

        return df_prepared
    

    def model_def(self) -> Pipeline:
        """
        define a model training pipeline object using SKlearn pipeline class

        class uses one hot encoding for categorical variables and instantiates 
        a random forest regressor
        """
        pipeline =Pipeline([
            ("enc", OneHotEncoder(variables=['main','description'],drop_last=True)),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        return pipeline
        
    def train(self, run_name="model_run"):
        logger.info("Starting training process...")

        df_prepared = self.preprocess()

        X = df_prepared.drop(columns=[self.target_col, 'date'])  # Drop target + datetime
        y = df_prepared[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model = self.model_def()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_type": type(self.model).__name__,
                "test_size": self.test_size,
                "random_state": self.random_state
            })
           
            self.model.fit(X_train, y_train)
            logger.info("Model training complete.")

            self.X_test, self.y_test = X_test, y_test

            self.evaluate()

        # Log model
            mlflow.sklearn.log_model(self.model, "model")

    def evaluate(self):
        logger.info("Evaluating model...")
        y_pred = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        mlflow.log_metrics({
            "mae": mae,
            "r2": r2
        })

        logger.info(f"MAE: {mae:.3f}")
        logger.info(f"R^2: {r2:.3f}")

        return {"mae": mae,  "r2": r2}