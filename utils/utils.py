# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import shutil
from tqdm import tqdm
import pydoc
import omegaconf


def flatten_dict(d, top_level_key="", sep="_"):
    """
    test = {"a": 1,
            "b": {"bb": 2},
            "c": {"cc": {"ccc":3}, "cc1": 33},
            "d": {"dd": {"dddd": {"dddd":4}}}}
    print(*[f"{k}: {v}" for k, v in utils.flatten_dict(cfg).items()], sep="\n")

    Args:
        d (_type_): _description_
        top_level_key (str, optional): _description_. Defaults to "".
        sep (str, optional): _description_. Defaults to "_".

    Returns:
        _type_: _description_
    """
    flat_d = {}
    for k, v in d.items():
        if isinstance(v, (dict, omegaconf.DictConfig)):
            flat_d.update(flatten_dict (v, top_level_key=(top_level_key + k + sep)))
        else:
            flat_d[top_level_key + k] = v
    return flat_d

def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = dict(d.copy())
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs) 

    return pydoc.locate(object_type)(**kwargs)
    

def merge_pdfs(base_path=".", individual_pdfs_folder="", pdf_path_list=None, out_file_name = "merged.pdf", remove_files=False):
    from PyPDF2 import PdfFileMerger

    if pdf_path_list is None:
        inputs_folder = os.path.join(base_path, individual_pdfs_folder)
        pdf_paths = glob.glob(os.path.join(inputs_folder, "**", "*.pdf"), recursive=True)
        if not pdf_paths: 
            # No pdfs to merge were found
            return
    else:
        inputs_folder = os.path.commonpath(pdf_path_list)
        pdf_paths = pdf_path_list
    
    merger = PdfFileMerger()

    for pdf in tqdm(pdf_paths):
        merger.append(pdf)
        if remove_files:
            os.remove(pdf)

    merged_pdf_path = os.path.join(base_path, out_file_name)
    merger.write(merged_pdf_path)
    merger.close()
    print("Merged pdf saved to: ", os.path.abspath(merged_pdf_path))

    if remove_files:
        print("Removing folder", inputs_folder)
        shutil.rmtree(inputs_folder)



def get_next_version(root_dir):
    existing_versions = []
    if not os.path.exists(root_dir):
        return 0

    for dir_name in os.listdir(root_dir):
        if dir_name.startswith("version_"):
            version_number = int(dir_name.split("_")[1])
            existing_versions.append(version_number)

    if not existing_versions:
        return 0

    return max(existing_versions) + 1



    

if __name__ == '__main__':
    test = {"a": 1,
            "b": {"bb": 2},
            "c": {"cc": {"ccc":3}, "cc1": 33},
            "d": {"dd": {"dddd": {"dddd":4}}}}

    print(*[f"{k}: {v}" for k, v in flatten_dict(test).items()], sep="\n")


