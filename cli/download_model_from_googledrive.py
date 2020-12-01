from utils.googledrive_requests import download_from_googledrive

file_id = "1QSRCuV0PBi20fIm1yblB4CWH5p_V6EVo"
filepath = "../data/models/resnet_model"

download_from_googledrive(file_id, filepath)

#TODO: upload kneenet to Gdrive and get file_id
file_id = "1QSRCuV0PBi20fIm1yblB4CWH5p_V6EVo"
filepath = "../data/models/KneeNet"
download_from_googledrive(file_id, filepath)