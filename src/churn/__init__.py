import logging 
import os 
import sys 
from datetime import datetime 

# Define the logfile name using the current date and time 
log_file_name = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"

# Create the path to the log file 
log_file_path = os.path.join(os.getcwd(), "logs", log_file_name)

# Create the log directory if it doesn't exist 
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Logger configuration 
logging.basicConfig(
    # Log format 
    format = "[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
    handlers=[
        logging.FileHandler(log_file_path), # Output logs to the file 
        logging.StreamHandler(sys.stdout) # Output logs in the terminal/console
    ]
)

logger = logging.getLogger('churn')