import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import ants
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import nibabel as nib

from data.utils import log_3d

MRI_BLACKLIST_ID = [
    'I200978', 'I11193011', 'I180185', 'I217885',  # no data
    'I11096353', 'I11088545',  # broken image
    'I32421', 'I32853', 'I74064'  # not brain mri
]
affine = np.eye(4)
mri_template = ants.image_read('template/stripped_cropped.nii')


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
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: x.InstanceNumber)
    image_3d = np.stack([s.pixel_array for s in slices])
    return image_3d.squeeze()


def skull_stripping(img, id, temp_dir):
    input_path = os.path.join(temp_dir, f"{id}.nii")
    output_path = os.path.join(temp_dir, f"{id}_stripped.nii")
    nifti_img = nib.Nifti1Image(img, affine)
    nib.save(nifti_img, input_path)
    command = f'docker run --rm --gpus all -v {temp_dir}:/temp freesurfer/synthstrip:1.6 -i /temp/{id}.nii -o /temp/{id}_stripped.nii'
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    if proc.returncode != 0:
        print(f'stripping failed: {proc.returncode}')
    return output_path
    # return nib.load(f"{id}_stripped.nii").get_fdata()


def mri_registration(path):
    moving_image = ants.image_read(path)
    moving_image = ants.n4_bias_field_correction(moving_image)
    with tempfile.TemporaryDirectory() as temp_dir:
        outprefix = os.path.join(temp_dir, "registration_")

        registration = ants.registration(
            fixed=mri_template,
            moving=moving_image,
            type_of_transform='Affine',
            outprefix=outprefix
        )

        return registration["warpedmovout"].numpy()


def z_normalize_image(img: np.ndarray) -> np.ndarray:
    img = np.maximum(img, 0)  # Clip negatives if any (common in raw MRI)
    mask = img > 0
    foreground = img[mask]
    p1, p99 = np.percentile(foreground, [1, 99])
    img = np.clip(img, p1, p99)
    foreground = img[mask]
    mean = np.mean(foreground)
    std = np.std(foreground)
    img = (img - mean) / std
    return img.astype(np.float32)


def process_mri(row):
    img_id = row['Image Data ID']
    path = row['path']
    img = read_image(path)
    img = np.transpose(img, (0, 2, 1))
    img = img[::-1, ::-1, ::-1]
    with tempfile.TemporaryDirectory() as temp_dir:
        stripped_path = skull_stripping(img, img_id, temp_dir)
        processed_mri = mri_registration(stripped_path)
    processed_mri = z_normalize_image(processed_mri)
    try:
        log_3d(processed_mri, file_name=f'log/mri/full_processed/{row["Image Data ID"]}')
        # log_3d(processed_mri, file_name=None)
    except:
        print(row['path'])
        print(img.shape)
    # return img.shape


def create_mri_dataset():
    df = pd.read_csv('dataset/mri/mri_path.csv')
    # for row in tqdm(df.itertuples(), total=len(df), leave=False):
    # for index, row in tqdm(df.iterrows(), total=len(df), leave=False):
    #     process_mri(row)
    # img = read_image(row.path)
    # img = read_image(row.path)
    # log_3d(img, file_name=f'log/mri/raw/{row["Image Data ID"]}')
    # with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    with ProcessPoolExecutor(max_workers=4) as executor:
        # shapes = list(tqdm(executor.map(process_mri, [row for _, row in df.iterrows()]), total=len(df)))
        list(tqdm(executor.map(process_mri, [row for _, row in df.iterrows()]), total=len(df)))
    # print(np.unique(shapes, axis=0, return_counts=True))


def run():
    # merge_mri_descriptions_csv()
    # mri_info('dataset/mri/ADNI')
    create_mri_dataset()
