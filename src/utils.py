from typing import Union, List
from pathlib import Path
from itertools import product


def create_folder_for_each_czi(folder_path: Union[str, Path],
                               created_dirs_path: Union[str, Path] = "./"):
    list_of_name = [fn.stem for fn in Path(folder_path).iterdir() if fn.suffix == ".czi"]
    for name in list_of_name:
        Path(created_dirs_path, name).mkdir(parents=True, exist_ok=True)
    return list_of_name


def list_all_folders(rootpath: str) -> List[Path]:
    well_list = ['A','B','C','D','E','F','G','H']
    folder_list = [Path(rootpath, f"{w}{j}") for w in well_list for j in range(1, 13)]
    return folder_list


def list_dir_tif(path, list_to_read, list_to_save,list_of_name):
    """
    Get all the tif file in a folder and return the file paths to save and the file stems
    """ 
    for file in os.listdir(path): 
        file_path = os.path.join(path, file) 
        if os.path.splitext(file_path)[1]=='.tif': 
            list_to_read.append(file_path)
            list_to_save.append((os.path.splitext(file_path)[0] +' mask.png'))
            list_of_name.append((file_path.split('/')[-1])[:-4])
    return list_to_read,list_to_save,list_of_name


def list_tif_in_dir(folder_path: Union[str, Path],
                    sel_levels) -> List[Path]:
    """
    Get all the tif file in a folder and return the tiff files' paths
    """ 
    if sel_levels is None:
        return list(Path(folder_path).rglob(".tif"))
    valid_dirs = [Path(d) for d in product(*sel_levels)]
    file_names = []
    for vdir in valid_dirs:
        file_names.extend(list(vdir.rglob(".tif")))
    return file_names