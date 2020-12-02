# airmex

### 1. Getting started
1. Make a Python 3.7 venv and install requirements.txt

### 2. Downloading files from Google Drive

1. Make a target folder structure for files downloaded from googledrive. Default structure is:
    - ./data
      - /models
      - /test
2. Run tests.testgoogledrive.io (preferably with pytest). This downloads a small text file as a test,
3. A script to download a Resnet model is in cli.download_model_from_googledrive.py
