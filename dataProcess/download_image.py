import argparse
import urllib.request as urllib2
import os.path
from bs4 import BeautifulSoup
import requests
import re
from pathlib import Path
import random
import sys
import torchvision.transforms as T
import requests
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.system import runLinuxCmd

toPIL = T.ToPILImage()

def count_lines(filename):
    with open(filename, 'r') as f:
        count = 0
        for line in f:
            count += 1
    return count


def split_str(s, n):
    length = len(s)
    return [ s[i:i+n] for i in range(0, length, n) ]

def img_download(url, filename):
    img = urllib2.urlopen(url)
    fout = open(filename, 'wb')
    fout.write(img.read())
    img.close()
    fout.close()
    

def download(args):

    os.makedirs(args.target_folder, exist_ok=True)
    totalLineNum = count_lines(Path(args.metadata_folder) / 'yfcc100m_dataset')
    validLines = random.sample(range(totalLineNum), args.sample_size)

    fin = open(Path(args.metadata_folder) / 'yfcc100m_dataset')
    

    print('Start downloading YFCC100M dataset...')
    i = 0
    img_count = 0
    for line in fin:
        if i in validLines:
            line_split = line.strip().split('\t')
            photo_id = line_split[0]    # photo id
            photo_url = line_split[13]    # photo URL for downloading
            photo_ext = os.path.splitext(photo_url)[1]

            # get the original image's url
            try:
                url_download = line_split[14] 
                guess_url = photo_url + 'sizes/s'
                htmldata = requests.get(guess_url).text
                soup = BeautifulSoup(htmldata, 'html.parser')
                scores = soup.find_all(text=re.compile('Original'))
                divs = [score.parent for score in scores]
                if len(divs) == 0: continue
                correct_url = photo_url + divs[0].get('href')[-8:]
            
                #get the image download url
                htmldata = requests.get(correct_url).text
                soup = BeautifulSoup(htmldata, 'html.parser')
                urls =  [item['src'] for item in soup.find_all('img')]
                
                for url in urls:
                    if url[-3:] == 'jpg':
                        url_download = url
                        # print(url_download)
                        runLinuxCmd(f"wget -c {url_download} -O {args.target_folder}/{photo_id}.jpg")
                        img_count += 1
                        if img_count % 100 == 0:
                            print(i,"Line finished.", "image count", img_count)
            except Exception as e:
                print("\tException occurs!")
                print("\t",e)
        i += 1
            

       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, default="path/to/target/folder")
    parser.add_argument("--metadata_folder", type=str, default="path/to/metadata/folder")
    parser.add_argument("--sample_size", type=int, default=2000000)
    args = parser.parse_args()
    
    download(args)

