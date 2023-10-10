import sys
import os.path
import os
import argparse
import json
from pathlib import Path
import urllib.parse

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.general import str_dict, num_dict




def extract_metadata(elems):
    if len(elems)<2: return None
    text = elems[1]
    text_split = text.split(',')
    d = str_dict()
    for elem in text_split:
        key = elem.split(':')[0]
        val = elem.split(':')[1]
        d[key] = val
    return d

import re
def refine_value(key, value):
    if key[-10:] == 'Resolution' or key == 'Exif+SubIFD.Exposure+Bias+Value':
        split_value = value.split()
        digits = split_value[0].split("/")
        try: 
            if len(digits)>1: split_value[0] = str(int(digits[0]) // int(digits[1]))
        except:
            split_value[0] = digits[0]
        value = " ".join(split_value)
    elif key == 'Exif+SubIFD.Focal+Length':
        split_value = value.split()
        split_value[0] = "{:.1f}".format(float(split_value[0]))
        value = " ".join(split_value)
    elif key == 'Exif+SubIFD.ISO+Speed+Ratings':
        split_value = value.split()
        value = split_value[0]
    elif len(value) > 20 : value = "Unknown"
    else: value = re.sub(r'[^a-zA-Z0-9.,/-]', '', value)
    return value
        
    
def url_decode(value):
    value =  urllib.parse.unquote(value)
    value = urllib.parse.unquote_plus(value)
    return value


def decode_exif_key(url_key):
    """decode url encoded exif keys and remove the prefix"""
    write_key = url_key.split(".")[1]
    write_key =  urllib.parse.unquote(write_key)
    write_key = urllib.parse.unquote_plus(write_key)
    return write_key
    
    


def download_exif_info(args):
    
    # if not args.debug: logger = Logger(name="metadata_download", config={"totalNum":args.total_number})
    
    fin_exif = open(args.metadata_path)
    
    line_count = 0
    image_count = 0
    all_exif_count = num_dict()
    
    while True:
        line_count += 1
        

        if image_count >= args.total_number: 
            print("About to break: (line_count, image_count)", line_count, image_count)
            break

        if line_count % 100 == 0: print(f"[line_count: {line_count}], [img_count: {image_count}]") 
        
        
        # extract photo id
        line_e = fin_exif.readline()
        line_e_split = line_e.strip().split('\t')
        photo_id = int(line_e_split[0])
        
        
        # check if photo path exists
        file_path = os.path.join(args.img_folder_path, f"{photo_id}.png")
        if not os.path.exists(file_path): continue
        
        
        # extract exif 
        exif = extract_metadata(line_e_split)
        
        # new_row = {'line': line_count, 'photo_id': photo_id}
        new_row = {"EXIF":{}, "photo_id": photo_id, "num_exif_tags": 0}
        if exif:
            if len(exif.keys())>10:
                image_count += 1
                for key in exif.keys():
                    if exif[key] =='None' and exif[key][:7] == "Unknown": continue
                    value = url_decode(exif[key])
                    write_key = decode_exif_key(key)
                    try:
                        value = refine_value(key, value)
                        new_row["EXIF"][ write_key ] = value
                        all_exif_count[ write_key ] += 1
                    except:
                        new_row["EXIF"][ write_key ] = 'Unknown'
                new_row["num_exif_tags"] = len(new_row["EXIF"].keys())
                

                json_object = json.dumps(new_row, indent = 4)
                with open(f"{args.img_folder_path}/{photo_id}.json", "w") as outfile:
                    outfile.write(json_object)
                    
    for k in all_exif_count.keys():
        all_exif_count[k] = all_exif_count[k] / image_count
        
    all_exif_count = dict(sorted(all_exif_count.items(), key=lambda item: item[1]))
    json_object = json.dumps(all_exif_count, indent = 4)
    with open(f"dataProcess/all_exif_count.json", "w") as outfile:
        outfile.write(json_object)
                
            
    
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", default="debug", help="name your experiment")
    parser.add_argument("--img_folder_path", type=str, default="path/to/img/folder")
    parser.add_argument("--total_number", type=int, default=2000000)

    args = parser.parse_args()
    
    download_exif_info(args)