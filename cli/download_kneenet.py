from utils.googledrive_requests import download_from_googledrive
import os
import dotenv
dotenv.load_dotenv()

"""KneeNet Download"""
file_id = os.getenv("KneeNet_file_id")
target_filepath = os.getenv("KneeNet_filepath")
download_from_googledrive(file_id, target_filepath)

"""Example input download"""
file_id = os.getenv("example_input_file_id")
target_filepath = os.getenv("example_input_filepath")
download_from_googledrive(file_id, target_filepath)

