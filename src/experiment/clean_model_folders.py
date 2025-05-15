#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to clean up model folders in results/cgcnn_models_opt.
For each task folder:
1. Sort version folders by version number
2. Keep the last version folder and others that contain test_metrics.csv
3. Rename remaining folders to have consecutive version numbers
"""

import os
import shutil
import re
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# Base directory to process
BASE_DIR = ROOT_DIR/'results/cgcnn_models_opt'


def get_version_number(folder: Path) -> int:
    """Extract version number from folder name 'version_X'."""
    folder_name = Path(folder).name
    match = re.match(r'version_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return -1


def clean_task_folder(task_folder: Path) -> None:
    """Clean a task folder by removing unnecessary version folders and renaming the rest."""
    logger.info(f"Processing task folder: {task_folder.name}")
    
    # Find all version folders
    version_folders = [f for f in task_folder.glob("version_*") if f.is_dir()]
    if not version_folders:
        logger.info(f"No version folders found in {task_folder.name}")
        return
    
    # Sort version folders by version number
    version_folders.sort(key=get_version_number)
    
    # Get the last version folder name
    last_version = version_folders[-1]
    
    # Collect folders to keep
    folders_to_keep = [last_version]
    
    # Check other folders for test_metrics.csv except the last one
    for folder in version_folders[:-1]:
        if (folder / 'test_metrics.csv').exists():
            folders_to_keep.append(folder)
        else:
            logger.info(f"Folder {folder} in {task_folder.name} will be removed (no test_metrics.csv)")
    
    # Sort the list of folders to keep
    folders_to_keep.sort(key=get_version_number)
    
    # Remove remaining version folders
    for folder in version_folders:
        if folder in folders_to_keep:
            continue
        if folder.exists():
            try:
                shutil.rmtree(folder)
                logger.info(f"Removed {folder.name} from {task_folder}")
            except Exception as e:
                logger.error(f"Error removing {folder}: {e}")
    i = 0
    name_map = {}
    name_map_file = task_folder / "name_map.json"
    if name_map_file.exists():
        with open(name_map_file, "r") as f:
            name_map = json.load(f)
    name_map_reverse = {v: k for k, v in name_map.items()}
    # Rename remaining folders to have consecutive version numbers
    for folder in folders_to_keep:
        new_name = f"version_{i}"
        new_folder_path = task_folder / new_name
        if folder != new_folder_path and (folder != last_version or (folder / 'test_metrics.csv').exists()):
            try:
                shutil.move(folder, new_folder_path)
                logger.info(f"Renamed {folder.name} to {new_name} in {task_folder}")
                if folder.name in name_map_reverse:
                    name_map[name_map_reverse[folder.name]] = new_name
                else:
                    name_map[folder.name] = new_name
            except Exception as e:
                logger.error(f"Error renaming {folder} to {new_name}: {e}")
        i += 1
    # save the name map to a file
    with open(name_map_file, "w") as f:
        json.dump(name_map, f, indent=4)


def main() -> None:
    """Main function to process all task folders."""
    logger.info(f"Starting cleaning process in {BASE_DIR}")
    
    # Get all task folders (directories only)
    task_folders = []
    for d in BASE_DIR.iterdir():  ## rand0, rand1, rand2
        if not d.is_dir():
            continue
        for sub_d in d.iterdir():  ## task_folder
            if not sub_d.is_dir():
               continue
            task_folders.append(sub_d)
    print("Number of task folders:", len(task_folders))
    # Process each task folder
    for task_folder in task_folders:
        try:
            clean_task_folder(task_folder)
        except Exception as e:
            logger.error(f"Error processing {task_folder.name}: {e}")
    
    logger.info("Cleaning process completed")


if __name__ == "__main__":
    main()
    # wk_dir = Path("/home/zhangsd/repos/MOFSNN/results/cgcnn_models_opt/rand0")
    # bk_dir = Path("/home/zhangsd/repos/MOFSNN/results/cgcnn_models_opt_backup/rand0")
    # for task_dir in wk_dir.iterdir():
    #     if not task_dir.is_dir():
    #         continue
    #     if task_dir.name == "TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_cgcnn_raw":
    #         continue
    #     shutil.rmtree(task_dir)
    #     shutil.copytree(bk_dir/task_dir.name, task_dir)

    #     print(f"Restored {task_dir.name} from backup")
