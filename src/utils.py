from typing import Union, List
from pathlib import Path
from itertools import product
from aicsimageio import AICSImage
from collections import namedtuple
from enum import Enum
import tifffile
import os
import numpy as np


def create_folder_for_each_czi(folder_path: Union[str, Path], dest_folder=None, target_czi=None) -> dict:
    dest_folder = dest_folder if dest_folder is not None else folder_path
    path_dic = {fn.stem: Path(dest_folder, fn.stem) 
                for fn in Path(folder_path).iterdir() if fn.suffix == ".czi"}
    if target_czi is not None:
        path_dic = {k: v for k, v in path_dic.items() if k in target_czi}

    for name, path in path_dic.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    return path_dic


def sep_czi_channels_to_tiff(czi_file_path, save_dir, id_format="{czi_name}{i:04}", axes="TCZYX"):
    img = AICSImage(czi_file_path)
    ch_idx = axes.index("C")
    for ci in range(img.shape[ch_idx]):
        sel_img = img.get_image_data(axes, C=ci)

        tifffile.imwrite(
            Path(save_dir, id_format.format(i=ci, czi_name=Path(czi_file_path).stem)).with_suffix(".tif"),
            sel_img,
            imagej=False,
            photometric='minisblack',
            metadata={'axes': axes},
        )


def split_czi_to_tiffs(folder_path, target_czi=None, dest_folder=None, sep_dir=True, id_format="{czi_name}{i:04}", axes="TCZYX"):
    if target_czi is None:
        all_czi_paths = [fn for fn in Path(folder_path).glob("*.czi")]
    else:
        all_czi_paths = [fn for fn in Path(folder_path).glob("*.czi") if fn.stem in target_czi]
    if sep_dir:
        dest_folders = create_folder_for_each_czi(folder_path=folder_path, dest_folder=dest_folder,
                                                  target_czi=target_czi)
        for czi_path in all_czi_paths:
            sep_czi_channels_to_tiff(czi_path, dest_folders[czi_path.stem], id_format=id_format, axes=axes)
        return dest_folders
    dest_folder = folder_path if dest_folder is None else dest_folder
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    for czi_path in all_czi_paths:
        sep_czi_channels_to_tiff(czi_path, dest_folder, id_format=id_format, axes=axes)
    return {"dest_folder": dest_folder}


def list_all_folders(rootpath: str) -> List[Path]:
    """
    A deprecated function.
    Get all paths created for all the well names
    """ 
    well_list = ['A','B','C','D','E','F','G','H']
    folder_list = [Path(rootpath, f"{w}{j}") for w in well_list for j in range(1, 13)]
    return folder_list


def list_dir_tif(path, list_to_read, list_to_save,list_of_name):
    """
    A deprecated function.
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
        return list(Path(folder_path).rglob("*.tif"))
    valid_dirs = [Path(folder_path) / Path(*d) for d in product(*sel_levels)]
    file_names = []
    print(f"Finding files from {valid_dirs}")
    for vdir in valid_dirs:
        file_names.extend(list(vdir.rglob("*.tif")))
    return file_names


class SearchDomain(Enum):
    uniform = "u"
    loguniform = "lu"
    categorical = "c"
    intuniform = "iu"


HyperParam = namedtuple("HyperParam",
                        ["name", "search_range", "search_domain"],)


def get_hparams_grid(*hparams, n, random_k=None, seed=42, step_like=True, **kwargs):
    assert [isinstance(hp, HyperParam) for hp in hparams], "Passed hparams must be HyperParams"
    rng = np.random.default_rng(seed=seed)
    hparam_choices = []
    hparam_order = []
    
    for hparam in hparams:
        hparam_order.append(hparam.name)
        low, high = hparam.search_range if hparam.search_domain != SearchDomain.categorical else (0, len(hparam.search_range))
            
        if hparam.search_domain == SearchDomain.categorical:
            choices = rng.choice(hparam.search_range, replace=False, size=(min(n, len(hparam.search_range)),))
        elif hparam.search_domain == SearchDomain.loguniform:
            choices = np.exp(rng.uniform(low=np.log(low), 
                                         high=np.log(high), 
                                         size=(n,))) if not step_like else np.logspace(np.log10(low), np.log10(high), n,
                                                                                       base=10)
        elif hparam.search_domain == SearchDomain.uniform:
            choices = rng.uniform(low=low, 
                                  high=high, size=(n,)) if not step_like else np.linspace(low, high, n)
        elif hparam.search_domain == SearchDomain.intuniform:
            step = (high - low) / (n - 1)
            choices = rng.integers(low=hparam.search_range[0], high=hparam.search_range[1], size=(n,)) \
                      if not step_like else np.arange(low, high+1, step).astype(int)
        hparam_choices.append(choices)
    hparam_sets = product(*hparam_choices)
    hparam_sets = [dict(zip(hparam_order, hset)) for hset in hparam_sets]
    if random_k is None:
        return hparam_sets
    
    assert random_k < np.power(n, len(hparams))
    
    return rng.choice(hparam_sets, size=(random_k,), replace=False)
