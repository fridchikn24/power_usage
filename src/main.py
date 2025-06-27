import pandas as pd
from model_pipeline.model_service import ModelService
from model_pipeline.collection import load_data_from_csv
from loguru import logger

@logger.catch 
def main():
    logger.info("running the application...")

    df = load_data_from_csv()
    ml_svc = ModelService(df)
    ml_svc.load_model()

    

if __name__ == '__main__':
    main()