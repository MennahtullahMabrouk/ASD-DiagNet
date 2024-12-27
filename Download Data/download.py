import os
import requests
import io
import pandas as pd
from nilearn.datasets import fetch_abide_pcp
import argparse

def fetch_file(url, output_path):
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}, skipping download.")
        return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Failed to fetch {url}: {response.status_code}")

import warnings
warnings.filterwarnings('ignore')

def download_file(url, output_path, verbose=0):
    """Custom function to download a file."""
    if os.path.exists(output_path):
        if verbose:
            print(f"File already exists: {output_path}, skipping download.")
        return

    if verbose:
        print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if verbose:
            print(f"Saved: {output_path}")
    else:
        if verbose:
            print(f"Failed to download: {url} (Status code: {response.status_code})")

def fetch_abide(root_dir='./data', data_type='func', preproc_ii=False,
                band_pass_filtering=False, global_signal_regression=False,
                pipeline='cpac', verbose=0):
    # Fetch ABIDE I
    abide1_type = [data_type + '_preproc']
    abide1 = fetch_abide_pcp(data_dir=root_dir, derivatives=abide1_type, verbose=verbose,
                             pipeline=pipeline, band_pass_filtering=band_pass_filtering,
                             global_signal_regression=global_signal_regression)

    abide1_num = len(abide1['func_preproc'])

    # Get the ABIDE I path
    strategy = 'no' if not band_pass_filtering else ''
    strategy += 'filt_'
    strategy += 'no' if not global_signal_regression else ''
    strategy += 'global'

    data_folder = 'ABIDE_pcp'
    path = os.path.join(root_dir, data_folder, pipeline, strategy)

    # Fetch ABIDE II (preprocessed fMRI is not available, fetch raw fMRI)
    base_url = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/RawData/'

    sites = ['USM_1', 'UCLA_1', 'UCD_1', 'TCD_1', 'SDSU_1',
             'ONRC_2', 'OHSU_1', 'NYU_2', 'NYU_1', 'KUL_3',
             'KKI_1', 'IU_1', 'IP_1', 'GU_1', 'ETHZ_1',
             'EMC_1', 'BNI_1']

    phenotypic = pd.DataFrame()

    if not preproc_ii:
        abide2_num = 0
        for site in sites:
            site_url = base_url + f'ABIDEII-{site}'
            phenotypic_url = site_url + '/participants.tsv'

            if requests.get(phenotypic_url).status_code == 404:
                print(f'Phenotypic data for {site} not found.')
                continue

            print(f'Successfully fetched phenotypic data for {site}')

            # Load phenotypic data
            temp_table = pd.read_csv(io.StringIO(requests.get(phenotypic_url).text), sep='\t')
            phenotypic = pd.concat([phenotypic, temp_table])

            # Download the files for each participant
            ids = temp_table['participant_id']
            print(f'{len(ids)} samples to download in {site} dataset')

            for participant_id in ids:
                child_path = f'/sub-{participant_id}/ses-1/func'
                file_name = f'/sub-{participant_id}_ses-1_task-rest_run-1_bold.nii.gz'
                file_url = site_url + child_path + file_name

                # Skip if the file doesn't exist
                if requests.get(file_url).status_code == 404:
                    print(f'No such file: {file_url}')
                    continue

                abide2_num += 1
                file_path = os.path.join(path, file_name.split('/')[-1])

                # Download the file
                download_file(file_url, file_path, verbose=verbose)

    phenotypic.to_csv(os.path.join(root_dir, data_folder, 'Phenotypic_ABIDE2.csv'), index=False)

    return abide1_num, abide2_num

def main(args):
    abide1_num, abide2_num = fetch_abide(root_dir=args.data_dir, verbose=1)
    print(f"ABIDE I: {abide1_num} files downloaded.")
    print(f"ABIDE II: {abide2_num} files downloaded.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default='./data', help='Path to store the data')
    args = parser.parse_args()

    main(args)
