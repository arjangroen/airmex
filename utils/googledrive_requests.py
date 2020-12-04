import gdown
import dotenv



def download_from_googledrive(file_id: str, target_filepath: str):
    """
    Download a file from googledrive.
    :param file_id: googledrive file id
    :param target_filepath: Where to store the file
    :return:
    """
    url = "https://drive.google.com/uc?id=" + file_id
    gdown.download(url, target_filepath, quiet=False)
