from utils.googledrive_requests import download_from_googledrive
import os
import dotenv
dotenv.load_dotenv()

"""
Chen, Pingjun (2018), “Knee Osteoarthritis Severity Grading Dataset”, 
Mendeley Data, V1, doi: 10.17632/56rmx5bjcr.1
https://data.mendeley.com/datasets/56rmx5bjcr/1
"""
file_id = os.getenv("mendeley_dataset_file_id")
target_filepath = "data/mendeley/KneeKL299.zip"
download_from_googledrive(file_id, target_filepath)
