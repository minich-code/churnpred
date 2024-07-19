from src.churn import logging
from src.churn.pipelines.pip_01_data_ingestion import DataIngestionPipeline
from src.churn.pipelines.pip_02_data_validation import DataValidationPipeline

COMPONENT_01_NAME = "DATA INGESTION COMPONENT"
try: 
    logging.info(f"# ====================== {COMPONENT_01_NAME} Started! ============================== #")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logging.info(f"# ====================== {COMPONENT_01_NAME} Terminated Successfully! ===============##\n\nx******************x")
except Exception as e:
    logging.exception(e)
    raise e

COMPONENT_02_NAME = "DATA VALIDATION COMPONENT"
try:
    logging.info(f"# ====================== {COMPONENT_02_NAME} Started! ================================= #")
    data_validation_pipeline = DataValidationPipeline()
    data_validation_pipeline.run()
    logging.info(f"## ======================== {COMPONENT_02_NAME} Terminated Successfully!=============== ##\n\nx************************x")

except Exception as e:
    logging.exception(e)
    raise e