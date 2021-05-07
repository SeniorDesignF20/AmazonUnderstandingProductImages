import os 
import json 
import gzip
import pandas as pd 
import urllib.request
import requests
import shutil
from PIL import Image, ImageOps
from pathlib import Path
import cv2
import sys

#create a 2d-list containing first-n digit id identical products(most cases)

product_dict = {}
def create_pdict(digit, current_cleaning_dir):
    global product_dict
    product_dict.clear()
    for file in os.listdir(current_cleaning_dir):
        key_id = file[:digit]
        if key_id not in product_dict:
            product_dict.setdefault(key_id,[]).append(file)
        else:
            product_dict[key_id].append(file)
    pdict_len = len(list(product_dict))        
    #print("first {} digit: ".format(str(digit)) + str(pdict_len))
    return pdict_len
    
def check_duplicates(p_list, current_cleaning_dir):
  final_list = []
  index_list = []
  to_write = set()
  for filename in p_list:
    index = filename[11:filename.rfind(".")]
    if index not in index_list:
      final_list.append(filename)
      index_list.append(index)
    else:  
      for f in final_list:
        if index == f[11:f.rfind(".")]:
          img1 = os.path.join(current_cleaning_dir, f)
          img2 = os.path.join(current_cleaning_dir, filename)
          try:
            assert(os.path.isfile(img1))
            assert(os.path.isfile(img2))
          except:
            display(final_list)
            display(index_list)
            display(to_write)
            display(f)
            display(filename)
            
          if open(img1,"rb").read() == open(img2,"rb").read():
            to_write.add(f)
            to_write.add(filename)
            os.remove(img2)
            break
          else:
            continue
      if filename not in to_write:
        final_list.append(filename)

  #for filename in to_write:
    #write_duplicates(filename)
    
def check_duplicates_all(p_list, current_cleaning_dir):
    final_list = []
    to_write = set()
    final_list.append(p_list[0])
    for filename in p_list:
        if filename not in final_list:
            for f in final_list:
                img1 = os.path.join(current_cleaning_dir, f)
                img2 = os.path.join(current_cleaning_dir, filename)
                if open(img1,"rb").read() == open(img2,"rb").read():
                    to_write.add(f)
                    to_write.add(filename)
                    os.remove(img2)
                    break
                else:
                    continue
            if filename not in to_write:
                final_list.append(filename)
    #for fn in to_write:
        #write_duplicates(filename)

digits = [6,4,9]
exceptions = ["__pycache__", ".ipynb_checkpoints","classifier_models"]
new_classes = ['Brush', 'Case', 'Lashes', 'Lotion', 'Mirror', 'Nail', 'Palette', 'Perfume', 'Shaver', 'Skincare', 'Spray', 'Tiara', 'Toothbrush', 'Wallets', 'Watch', 'Wig']
rootdir = os.path.join('..', 'ds')

for root, dirs, files in os.walk(rootdir):
    for current_dir in dirs:
        if current_dir in exceptions:
            continue
        if current_dir not in new_classes:
            continue
        current_class = current_dir
        print("Cleaning Class: "+current_class)
        current_cleaning_dir = os.path.join(rootdir, current_dir)
        
        
        for digit in digits:
            cnt = 0
            pdict_len = create_pdict(digit, current_cleaning_dir)

            for key in product_dict:
                p_list = product_dict[key]
                if digit == 9:
                    if len(p_list) > 1:
                        check_duplicates_all(p_list, current_cleaning_dir)
                else:        
                    asin_list = []
                    for file in p_list:
                        asin = file[:10]
                        if asin not in asin_list:
                            asin_list.append(asin)
                    if len(asin_list) > 1:
                        check_duplicates(p_list, current_cleaning_dir)

                cnt += 1
                sys.stdout.write("\rCleaning first{0} digits: {1}%".format(str(digit), (float(cnt)/pdict_len)*100))
            print('')            