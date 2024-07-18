import sys 
from src.churn import logging

# Method to extract error message detail 
def error_message_detail(error, error_details_object: sys):
    # Get the traceback information from error detail 
    _, _, exc_tb = error_details_object.exc.info()

    # Extract the filename from the traceback 
    file_name = exc_tb.tb_frame.f_code.co_file_name

    # Create a formatted error message 
    formatted_error_message = f"File: {{file_name}} \nLine Number: [{exc_tb.tb_lineno}] \nError message: [{error}]"
    
    
    return formatted_error_message

# Define the custom exception 
class FileOperationError(Exception):
    def __init__(self, formatted_error_message, error_details_object: sys):
        # Call a class constructor to set the error message 
        super().__init__(formatted_error_message)
        self.formatted_error_message = error_message_detail(formatted_error_message, error_details_object=error_details_object)


    # override the __str__ method to return the formatted error message
    def __str__(self):
        return self.formatted_error_message