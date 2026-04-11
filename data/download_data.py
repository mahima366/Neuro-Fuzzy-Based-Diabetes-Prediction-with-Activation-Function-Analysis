"""
Download the Pima Indians Diabetes Dataset
Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
"""
import os
import urllib.request

# Column names for the dataset
COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Public mirror URL (same data as Kaggle)
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def download_dataset():
    """Download the diabetes dataset CSV file."""
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "diabetes.csv")

    if os.path.exists(save_path):
        print(f"  Dataset already exists at: {save_path}")
        return save_path

    print(f"  Downloading dataset...")
    try:
        urllib.request.urlretrieve(DATASET_URL, save_path)
        # Add header row to the downloaded CSV
        with open(save_path, 'r') as f:
            content = f.read()
        with open(save_path, 'w') as f:
            f.write(','.join(COLUMNS) + '\n' + content)
        print(f"  Dataset saved to: {save_path}")
        return save_path
    except Exception as e:
        print(f"  Download failed: {e}")
        print("\n  Please download manually from:")
        print("    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        print(f"  Save the CSV file as: {save_path}")
        return None


if __name__ == "__main__":
    download_dataset()
