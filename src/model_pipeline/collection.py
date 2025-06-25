
import pandas as pd
from dotenv import load_dotenv
import os
from loguru import logger


load_dotenv()

def load_data_from_csv(path=os.getenv('DATA_FILE_NAME')):
    logger.info("extracting the table from the csv")
    return pd.read_csv(path)