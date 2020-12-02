import os
from pathlib import Path
from utils.googledrive_requests import download_from_googledrive


def test_download_from_googledrive():
    root_dir = Path(__file__).parent.parent
    target_filepath = os.path.join(root_dir, "data/test/test.txt")
    file_id = "11UTrpJymE4Af3k_8X6Bj1dBDkPnjRIdT"
    download_from_googledrive(file_id, target_filepath)
    with open(target_filepath, "r") as testfile:
        result = testfile.read()
        print("\n", result)
        assert result == "Succes!\n"
