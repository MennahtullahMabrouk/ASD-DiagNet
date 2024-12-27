import warnings
warnings.filterwarnings('ignore')

import requests
import os
import io
import pandas as pd
from nilearn.datasets import fetch_abide_pcp
import argparse
import time

def _fetch_file(url, file_path, verbose=0):
    """Helper function to download a file from a URL."""
    temp_file = file_path + ".part"
    try:
        # Ensure the directory exists before starting the download
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if os.path.exists(temp_file):
            os.rename(temp_file, file_path)
            if verbose:
                print(f"File downloaded and saved to {file_path}")
        else:
            raise FileNotFoundError(f"Temporary file {temp_file} not found after download.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e
    except Exception as e:
        print(f"Unexpected error while handling {url}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def _fetch_file_with_retries(url, file_path, retries=3, verbose=0):
    """Retries downloading a file multiple times if needed."""
    for attempt in range(retries):
        try:
            _fetch_file(url, file_path, verbose)
            return
        except FileNotFoundError as e:
            print(f"File not found error on attempt {attempt + 1}: {e}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(5)  # Wait before retrying
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts.")

def fetch_abide(root_dir='./data', data_type='func', preproc_ii=False,
                band_pass_filtering=False, global_signal_regression=False,
                pipeline='cpac', verbose=0):
    # Create the root directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)

    # Fetch ABIDE I
    abide1_type = [data_type + '_preproc']
    abide1 = fetch_abide_pcp(data_dir=root_dir, derivatives=abide1_type, verbose=verbose,
                             pipeline=pipeline, band_pass_filtering=band_pass_filtering,
                             global_signal_regression=global_signal_regression)

    abide1_num = len(abide1['func_preproc'])

    # Define path structure
    strategy = ''
    if not band_pass_filtering:
        strategy += 'no'
    strategy += 'filt_'
    if not global_signal_regression:
        strategy += 'no'
    strategy += 'global'

    data_folder = 'ABIDE_pcp'
    path = os.path.join(root_dir, data_folder, pipeline, strategy)

    # Fetch ABIDE II
    base_url = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/RawData/'

    ins = ['USM_1', 'UCLA_1', 'UCD_1', 'TCD_1', 'SDSU_1',
           'ONRC_2', 'OHSU_1', 'NYU_2', 'NYU_1', 'KUL_3',
           'KKI_1', 'IU_1', 'IP_1', 'GU_1', 'ETHZ_1',
           'EMC_1', 'BNI_1']

    phenotypic = pd.DataFrame()
    abide2_num = 0

    if not preproc_ii:
        for enum in ins:
            child_base = base_url + 'ABIDEII-' + enum
            file_name = '/participants.tsv'
            child_url = child_base + file_name

            try:
                response = requests.get(child_url)
                if response.status_code == 404:
                    print(f'Phenotypic data of {enum} not found at {child_url}')
                    continue

                print(f'Successfully fetched phenotypic data of {enum}')
                temp_table = pd.read_csv(io.StringIO(response.text), sep='\t')
                phenotypic = pd.concat([phenotypic, temp_table])

                ids = temp_table['participant_id']
                print(f'{len(ids.values)} samples to download in {enum} dataset')
                for id in ids:
                    child_path = f'/sub-{id}/ses-1/func'
                    file_name = f'/sub-{id}_ses-1_task-rest_run-1_bold.nii.gz'
                    child_url = child_base + child_path + file_name

                    file_name = child_url.split('/')[-1]
                    file_path = os.path.join(path, file_name)

                    if os.path.exists(file_path):
                        if verbose:
                            print(f"File {file_name} already exists.")
                        continue

                    try:
                        _fetch_file_with_retries(child_url, file_path, verbose=verbose)
                        abide2_num += 1
                    except Exception as e:
                        print(f"Error fetching file {child_url}: {e}")
            except Exception as e:
                print(f"Error processing {enum}: {e}")

    phenotypic.to_csv(os.path.join(root_dir, data_folder, 'Phenotypic_ABIDE2.csv'), index=False)
    return abide1_num, abide2_num

def main(args):
    data_dir = args.data_dir if args.data_dir else './data'
    fetch_abide(root_dir=data_dir, verbose=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data', help='Path to store the data')
    args, unknown = parser.parse_known_args()
    main(args)
