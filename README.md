````markdown
# Unsafe Following Distance Detection

This project is a computer vision system designed to detect vehicles, track them, and analyze their following distance to identify and flag unsafe driving behavior in real-time. It uses a deep learning object detector and a tracking algorithm to calculate Time-to-Collision (TTC) from video footage.

## Getting Started

Follow these steps to set up the project on your local machine.

### Prerequisites

* Python 3.8+
* Git

### 1. Clone the Repository

First, clone this repository to your local machine using the following command:

```bash
git clone [https://github.com/ChisengaSol/unsafe_following_distance.git](https://github.com/ChisengaSol/unsafe_following_distance.git)
cd unsafe_following_distance

### 2\. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 4\. Download the Dataset

This project uses the ["Vehicle Detection - 8 Classes"](https://www.kaggle.com/datasets/sakshamjn/vehicle-detection-8-classes-object-detection/data) dataset from Kaggle. You will need a Kaggle account and an API token to download it.

1.  **Get your Kaggle API Token:** Go to your Kaggle account page, and click on **"Create New API Token"**. This will download a `kaggle.json` file.

2.  **Place the Token:** Move the downloaded `kaggle.json` file into a folder named `.kaggle` inside your user's home directory (e.g., `C:\Users\<YourUsername>\.kaggle\`). This is the standard location the Kaggle API will look for credentials.

3.  **Run the Download Script:** Execute the following command from the project's root directory to download and unzip the data into the correct location (`data/vehicle_data_raw/`).

    ```bash
    python download_data.py
    ```

## Usage

The primary notebook for data exploration and model training is located at `notebooks/01_data_explore_and_model_train.ipynb`.

### Running the Exploration Notebook

1.  **Launch Jupyter Notebook:** From the project's root directory (with your virtual environment activated), start Jupyter:

    ```bash
    jupyter notebook
    ```

2.  **Open the Notebook:** Navigate to `notebooks/` and open `01_data_explore_and_model_train.ipynb`.

3.  **Run the Cells:** Execute the cells in the notebook sequentially.

**Important:** The notebook is configured to run from the project's root directory. The data path is set to `data/vehicle_data_raw/`. If you followed the setup steps correctly, this path will work without any changes. The notebook will guide you through:

  * Verifying the dataset structure.
  * Visualizing raw images and their labels.
  * Splitting the data into training and validation sets.
  * Visualizing the augmented data to confirm the preprocessing pipeline is working correctly.

<!-- end list -->

```
```
