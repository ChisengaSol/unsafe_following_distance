import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# NOTE: The dataset is downloaded from https://www.kaggle.com/datasets/sakshamjn/vehicle-detection-8-classes-object-detection/data


def download_and_unzip_dataset():
    """
    Downloads and unzips the vehicle detection dataset from Kaggle
    into the 'data/' directory.
    """

    dataset_slug = "sakshamjn/vehicle-detection-8-classes-object-detection"
    download_path = "data"
    unzip_path = os.path.join(download_path, "vehicle_data_raw")

    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Downloading dataset '{dataset_slug}' to '{download_path}'...")
    api.dataset_download_files(dataset_slug, path=download_path, quiet=False)

    # unzip
    zip_file_name = f"{os.path.basename(dataset_slug)}.zip"
    zip_path = os.path.join(download_path, zip_file_name)

    print(f"\nUnzipping '{zip_path}' to '{unzip_path}'...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(unzip_path)

    print("Cleaning up downloaded zip file...")
    os.remove(zip_path)

    print("\nDataset downloaded!")
    print(f"   Located in: {unzip_path}")


if __name__ == "__main__":
    download_and_unzip_dataset()
