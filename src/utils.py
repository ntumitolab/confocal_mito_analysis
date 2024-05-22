from typing import Union
from pathlib import Path


def create_folder_for_each_czi(folder_path: Union[str, Path],
                               created_dirs_path: Union[str, Path] = "./"):
    list_of_name = [fn.stem for fn in Path(folder_path).iterdir() if fn.suffix == ".czi"]
    for name in list_of_name:
        Path(created_dirs_path, name).mkdir(parents=True, exist_ok=True)
    return list_of_name
