import os
import tarfile
import subprocess
import json
from tqdm import tqdm

def make_tar_from_folder(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        
        
def file2tar(output_filename, source_dir, sizeLimit=None):
    with tarfile.open(output_filename, 'w:gz') as tar:
        count = 0
        for f in tqdm(os.listdir(source_dir), desc="file_taring"):
            tar.add(os.path.join(source_dir,f), arcname=f)
            os.remove(os.path.join(source_dir,f))
            count += 1
            if sizeLimit!=None and count >= sizeLimit: break
    tar.close()


def runLinuxCmd(cmd: str):
    arch = subprocess.check_output(cmd,shell=True) #type is byte
    return arch

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in tqdm(it, desc="Calculating"):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total
    
def get_file_num(path):
    count = 0
    for p in os.listdir(path):
        # check if current path is a file
        if os.path.isfile(os.path.join(path, p)):
            count += 1
    return count
# du -h --max-depth=1


def read_filename_from_tar(tar_path):
    tar = tarfile.open(tar_path)
    for member in tar.getmembers():
        yield member.name
        
def untar(tarpath, savepath):
    my_tar = tarfile.open(tarpath)
    my_tar.extractall(savepath) # specify which folder to extract to
    my_tar.close()
    
def remove_dir(dirpath):
    for filename in os.listdir(dirpath):
        fpath = os.path.join(dirpath, filename)
        try: os.remove(fpath)
        except Exception as e: print(e)
        
def read_json_file(filepath):
    f = open(filepath)
    data = json.load(f)
    f.close()
    return data