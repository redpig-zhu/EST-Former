# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re

import numpy as np

_EEG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EEG_DIR, ".."))
# 兰州 2015 ERP 128 通道 .raw；可用 --eeg-raw-dir 覆盖
_DEFAULT_RAW = os.path.join(_EEG_DIR, "EEG_128channels_ERP_lanzhou_2015")
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "multimodal_overlap", "Numpydataset")


def extract_subject_id(filename: str):
    """从文件名中提取 subject id（前 8 位数字），如 02010002erp 20150416 1131.raw"""
    match = re.match(r"(\d{8})", filename)
    if match:
        return match.group(1)
    return None


def read_raw_eeg_file(filepath, n_channels=128, dtype=np.float32, header_bytes=0, order="C"):
    """
    读取 .raw：
    - dtype/header_bytes 需与采集格式一致
    - 返回 (n_channels, n_samples)
    """
    with open(filepath, "rb") as f:
        if header_bytes > 0:
            f.read(header_bytes)
        raw = f.read()

    data = np.frombuffer(raw, dtype=dtype)

    n_samples = len(data) // n_channels
    usable = n_samples * n_channels
    if usable == 0:
        raise ValueError(f"No usable samples in {filepath}, check dtype/header_bytes/n_channels")

    if usable != len(data):
        data = data[:usable]

    eeg = data.reshape((n_samples, n_channels), order=order).T
    return eeg


def parse_args():
    p = argparse.ArgumentParser(description="EEG .raw -> Numpydataset/*.npy (subject-level)")
    p.add_argument(
        "--eeg-raw-dir",
        type=str,
        default=_DEFAULT_RAW,
        help=f"原始 .raw 所在目录（默认: {_DEFAULT_RAW}）",
    )
    p.add_argument(
        "--npy-out-root",
        type=str,
        default=_DEFAULT_OUT,
        help=f"输出根目录，其下创建 MDD/ 与 Control/（默认: {_DEFAULT_OUT}）",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标 <subject_id>.npy 已存在则跳过（默认关闭：与原始脚本一致，每次覆盖重写）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    datapath = os.path.abspath(args.eeg_raw_dir)
    base_npy = os.path.abspath(args.npy_out_root)
    save_path_mdd = os.path.join(base_npy, "MDD")
    save_path_control = os.path.join(base_npy, "Control")
    os.makedirs(save_path_mdd, exist_ok=True)
    os.makedirs(save_path_control, exist_ok=True)

    if not os.path.isdir(datapath):
        raise FileNotFoundError(
            f"EEG 原始目录不存在: {datapath}\n"
            f"请用 --eeg-raw-dir 指定实际 .raw 目录。"
        )

    lstpath = [f for f in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, f))]
    print(f"EEG raw dir: {datapath}")
    print(f"NPY out root: {base_npy}")
    print(f"Total EEG files found: {len(lstpath)}")
    if lstpath:
        print(f"Sample files: {lstpath[:5]}")

    mdd_list = [f for f in lstpath if f.startswith("0201")]
    control_list_0202 = [f for f in lstpath if f.startswith("0202")]
    control_list_0203 = [f for f in lstpath if f.startswith("0203")]
    control_total = control_list_0202 + control_list_0203

    print(f"\nMDD files: {len(mdd_list)}")
    print(f"Control files (0202): {len(control_list_0202)}")
    print(f"Control files (0203): {len(control_list_0203)}")
    print(f"Total Control files: {len(control_total)}")

    written = 0
    skipped_exists = 0
    skip_existing = bool(args.skip_existing)

    for label_name, file_list, save_dir in (
        ("MDD", mdd_list, save_path_mdd),
        ("Control", control_total, save_path_control),
    ):
        print(f"\nProcessing {label_name} files...")
        for eeg_file in file_list:
            filepath = os.path.join(datapath, eeg_file)
            print(f"Processing: {eeg_file}")

            subject_id = extract_subject_id(eeg_file)
            if not subject_id:
                print(f"  Warning: Could not extract subject id from {eeg_file}")
                continue

            save_filename = f"{subject_id}.npy"
            save_filepath = os.path.join(save_dir, save_filename)
            if skip_existing and os.path.isfile(save_filepath):
                skipped_exists += 1
                print(f"  Skip (exists): {save_filename}")
                continue

            try:
                eeg_data = read_raw_eeg_file(filepath)
            except Exception as e:
                print(f"  Failed to read: {e}")
                continue

            np.save(save_filepath, eeg_data)
            written += 1
            print(f"  Saved: {save_filename}, Shape: {eeg_data.shape}")

    print("\n" + "=" * 50)
    print("Processing completed!")
    if skip_existing:
        print(f"Saved: {written} | skipped (already exists): {skipped_exists}")
    else:
        print(f"Saved (overwrite if existed): {written}")
    print(f"Total MDD files listed: {len(mdd_list)}")
    print(f"Total Control files listed: {len(control_total)}")

    mdd_saved = os.listdir(save_path_mdd) if os.path.exists(save_path_mdd) else []
    control_saved = os.listdir(save_path_control) if os.path.exists(save_path_control) else []

    print(f"\nMDD dir file count: {len(mdd_saved)}")
    print(f"Control dir file count: {len(control_saved)}")

    if mdd_saved:
        print(f"MDD sample: {mdd_saved[:3]}")
        sample_file = os.path.join(save_path_mdd, mdd_saved[0])
        sample_data = np.load(sample_file)
        print(f"Sample shape: {sample_data.shape}")

    if control_saved:
        print(f"Control sample: {control_saved[:3]}")


if __name__ == "__main__":
    main()
