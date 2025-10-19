import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from data.utils import log_3d

MRI_BLACKLIST_ID = ['I200978', 'I11193011', 'I180185', 'I217885']


def merge_mri_descriptions_csv():
    mpr = pd.read_csv('dataset/mri/MPRAGE.csv')
    acc = pd.read_csv('dataset/mri/Accelerated.csv')
    msv = pd.read_csv('dataset/mri/msv.csv')
    df_list = [mpr, acc, msv]

    combined_df = pd.concat(df_list, ignore_index=True)

    # Save result
    # combined_df.to_csv("mri.csv", index=False)
    print(f"Combined {len(df_list)} files into combined_data.csv ({len(combined_df)} rows)")
    print(f"unique subjects: {len(combined_df['Subject'].unique())}")


def mri_info(mri_path):
    df = pd.read_csv('dataset/mri/mri.csv')
    df['path'] = None
    total_images = 0

    subjects = os.listdir(mri_path)
    for subject in tqdm(subjects, leave=False):
        descs = os.listdir(f"{mri_path}/{subject}")
        for desc in tqdm(descs, leave=False):
            dates = os.listdir(f"{mri_path}/{subject}/{desc}")
            for date in tqdm(dates, leave=False):
                img_ids = os.listdir(f"{mri_path}/{subject}/{desc}/{date}")
                for img_id in img_ids:
                    total_images += 1
                    if img_id in MRI_BLACKLIST_ID:
                        continue
                    # print(img_id)
                    path = f"{mri_path}/{subject}/{desc}/{date}/{img_id}"
                    df.loc[df['Image Data ID'] == img_id, 'path'] = path
                    matches = df[df['Image Data ID'] == img_id]
                    # assert len(matches) == 1, f"❌ {img_id}: {len(matches)} rows!"
                    print(f"❌ {img_id}: {len(matches)} rows!") if len(matches) != 1 else None
                    df.loc[matches.index, 'path'] = path

    print(f"Total images: {total_images}")
    print(f'all rows: {len(df)}')
    df = df.dropna(subset=['path'])
    print(f'rows with image: {len(df)}')
    print(f"final unique subjects: {len(df['Subject'].unique())}")
    df.to_csv("mri_path.csv", index=False)


def read_image(path: str) -> np.ndarray:
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm')]
    # files = sorted(glob.glob(f"{path}/*.dcm"),
    #                key=lambda f: pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: x.InstanceNumber)
    image_3d = np.stack([s.pixel_array for s in slices])
    return image_3d.squeeze()


def process_mri(row):
    img = read_image(row['path'])
    log_3d(img, file_name=f'log/mri/raw/{row["Image Data ID"]}')


def create_mri_dataset():
    df = pd.read_csv('dataset/mri/mri_path.csv')
    # for index, row in tqdm(df.iterrows(), total=len(df), leave=False):
    #     img = read_image(row['path'])
    #     log_3d(img, file_name=f'log/mri/raw/{row["Image Data ID"]}')
    # with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    with ProcessPoolExecutor(max_workers=4) as executor:
        tqdm(executor.map(process_mri, [row for _, row in df.iterrows()]), total=len(df))


def run():
    # merge_mri_descriptions_csv()
    # mri_info('dataset/mri/ADNI')
    create_mri_dataset()
