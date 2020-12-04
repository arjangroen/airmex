import os
from utils.googledrive_requests import download_from_googledrive
import dotenv
dotenv.load_dotenv()

def test_download_from_googledrive():
    file_id = os.getenv('test_file_id')
    target_filepath = os.getenv('test_filepath')
    download_from_googledrive(file_id, target_filepath)
    with open(target_filepath, "r") as testfile:
        result = testfile.read()
        print("\n", result)
        assert result == "Succes!\n"