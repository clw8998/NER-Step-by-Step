from huggingface_hub import hf_hub_download
import os
import shutil
import zipfile

# Function to unzip the file and delete the zip file
def unzip_and_remove(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())
        print(f"Unzipped {zip_file_path} to current directory.")
    os.remove(zip_file_path)
    print(f"Deleted zip file: {zip_file_path}")

# Check if the unzipped folder exists before downloading``
def download_if_not_exists(repo_id, filename, folder_name):
    if os.path.exists(folder_name):
        print(f"Folder {folder_name} already exists. Skipping download.")
        return None
    else:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

# Download random_samples_1M.zip if it doesn't exist in the current directory
random_samples_1M_path = download_if_not_exists(
    repo_id="clw8998/Semantic-Search-dataset-for-EE5327701", 
    filename="random_samples_1M.zip", 
    folder_name='./random_samples_1M'
)

# Unzip random_samples_1M.zip if it was downloaded
if random_samples_1M_path:
    unzip_and_remove(random_samples_1M_path)
