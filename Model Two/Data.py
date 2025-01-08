import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

# Define the download URLs for the ABIDE dataset
CPAC_PIPELINE_URL = "http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_Release_1.0/Outputs/cpac/filt_global/"  # C-PAC pipeline
PHENOTYPIC_DATA_URL = "http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_Release_1.0/Phenotypic_V1_0b.csv"  # Phenotypic data

# Define the local directory to save the data
SAVE_DIR = "./abide_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_file(url, save_path):
    """
    Downloads a file from a given URL and saves it to the specified path.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB

    with open(save_path, "wb") as file, tqdm(
        desc=save_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

def get_file_list(url):
    """
    Fetches the list of files available in a directory using HTTP.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch file list from {url}")

    # Parse the HTML content to extract file names
    soup = BeautifulSoup(response.text, "html.parser")
    file_links = [a["href"] for a in soup.find_all("a") if a["href"].endswith(".1D")]  # Filter for .1D files
    return file_links

def download_abide_data():
    """
    Downloads the ABIDE dataset (C-PAC pipeline) and phenotypic data.
    """
    print("Downloading ABIDE dataset...")

    # Download phenotypic data
    phenotypic_save_path = os.path.join(SAVE_DIR, "Phenotypic_V1_0b.csv")
    if not os.path.exists(phenotypic_save_path):
        print(f"Downloading phenotypic data to {phenotypic_save_path}...")
        download_file(PHENOTYPIC_DATA_URL, phenotypic_save_path)
    else:
        print(f"Phenotypic data already exists at {phenotypic_save_path}.")

    # Download functional data (C-PAC pipeline)
    functional_save_dir = os.path.join(SAVE_DIR, "functionals", "cpac", "filt_global", "rois_cc200")
    os.makedirs(functional_save_dir, exist_ok=True)

    # Fetch the list of files to download
    print("Fetching list of functional data files...")
    file_list = get_file_list(CPAC_PIPELINE_URL)
    print(f"Found {len(file_list)} files to download.")

    # Download all files
    for file_name in file_list:
        file_url = CPAC_PIPELINE_URL + file_name
        file_save_path = os.path.join(functional_save_dir, file_name)
        if not os.path.exists(file_save_path):
            print(f"Downloading {file_name} to {file_save_path}...")
            download_file(file_url, file_save_path)
        else:
            print(f"{file_name} already exists at {file_save_path}.")

    print("Download complete!")

if __name__ == "__main__":
    download_abide_data()