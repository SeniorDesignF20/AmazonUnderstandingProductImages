import os 
import json
import pandas as pd 
import shutil
from PIL import Image, ImageOps
from pathlib import Path
import cv2
import sys
import copy

#creates product subclass dictionary for target directory, with same first #digit asin
def create_pdict(digit, target_dir):
    global product_dict
    product_dict.clear()
    for file in os.listdir(target_dir):
        key_id = file[:digit]
        if key_id not in product_dict:
            product_dict.setdefault(key_id,[]).append(file)
        else:
            product_dict[key_id].append(file)
    pdict_len = len(product_dict)        
    print("class number of identical first {} digit: ".format(str(digit)) + str(pdict_len)+ " keys.")

#merge keys in product_dict to a new dictionary, with same first #digit asin as keys
#used for products with asin differences in last few digits
def process_dict(digit):
    new_dict = {}
    global product_dict
    for key in product_dict:
        #img_list = product_dict.pop(key)
        img_list = product_dict[key]
        new_key = key[:digit]
        if new_key not in new_dict:
            new_dict.setdefault(new_key,[])
        new_dict[new_key] += img_list
        
    product_dict = copy.deepcopy(new_dict)
    
#process proonly merge the subkeys with 1 ascii diff on #(digit+1) digit
#used for more aggresive merge
def process_dict_safe(digit):
    new_dict = {}
    global product_dict
    global safe_index_diff
    #process_dict(digit)
    
    ##create temp dict based on current #digit+1# keys
    dict_temp = {}
    for key in product_dict:
        img_list = product_dict[key]
        parent_key = key[:digit]
        if parent_key not in dict_temp:
            dict_temp.setdefault(parent_key,{})
        dict_temp[parent_key][key] = img_list
    
    
    ##loop through dict with #digit# keys
    for key in dict_temp:
        subkey_list = list(dict_temp[key].keys())
        subkey_list.sort()
        visited = []
        
        for new_key in subkey_list:
            if new_key in visited:
                continue
            visited.append(new_key)
            
            if new_key not in new_dict:
                new_dict.setdefault(new_key,[])
            new_dict[new_key] += product_dict[new_key]
            
            if subkey_list.index(new_key) < len(subkey_list)-1:
                next_key = subkey_list[subkey_list.index(new_key)+1]
                if abs(ord(next_key[digit])-ord(new_key[digit])) <= safe_index_diff:
                    visited.append(next_key)
                    new_dict[new_key] += product_dict[next_key]
    
    product_dict = copy.deepcopy(new_dict)       
        
#refering to the given pdict, merge the class given by cur_dir    
def merge_images(pdict, cur_dir):
    #print(pdict)
    cur_path = os.path.join(".", cur_dir)
    for key in pdict:
        p_list = pdict[key]
        key_path = os.path.join('.', cur_dir, key)
        Path(key_path).mkdir(parents=True, exist_ok=True)
        for file in p_list:
            if file.endswith('.jpg'):
                shutil.move(os.path.join(cur_path, file), os.path.join(key_path, file))
            
    print(r"Merge done for {}".format(cur_dir))
    
#release all merges
def release_all_merge():
    for root, dirs, files in os.walk('.'):
        for cur_dir in dirs:
            release_merge(cur_dir)
    print("All Merge released.")
    
#release merge in target class            
def release_merge(target_dir):
    for root, dirs, files in os.walk(target_dir):
        for subclass in dirs:
            for filename in subclass:
                shutil.move(os.path.join(target_dir, subclass, filename), os.path.join(target_dir, filename))
    print("Merge released in {}".format(target_dir))


product_dict = {}
raw_dict = {}
compress_dict = {}
digits = [8, 7, 6, 5, 4, 3]
exceptions = ['classifier_models', '.ipynb_checkpoints', 'duplicate ids', '.Trash']

#compress all classes under Dataset/, the classes should be uncompressed before calling
def merge(safe_digit_threhold=5, safe_index_diff = 1, compression_max = 75):
    for root, dirs, files in os.walk('.'):
        for cur_dir in dirs:
            if cur_dir in exceptions:
                continue
            print("Processing Class: " + cur_dir)
            cur_path = os.path.join(root, cur_dir)

            #count asin number of the class before compression
            raw_dict.setdefault(cur_dir, 0)
            create_pdict(10, cur_path)
            raw_dict[cur_dir] = len(product_dict)

            for digit in digits:
                if digit!=8 and compress_dict[digit+1][cur_dir] > compression_max:
                    print(r"Skip Compression of {0} merging last {1} digit.".format(cur_dir, str(10-digit)))
                    if digit not in compress_dict:
                        compress_dict.setdefault(digit,{})[cur_dir] = compress_dict[digit+1][cur_dir]
                    else:
                        compress_dict[digit][cur_dir] = compress_dict[digit+1][cur_dir]
                    continue

                if digit > safe_digit_threhold:
                    process_dict(digit)
                else:
                    process_dict_safe(digit)

                compress_rate = float(100*(raw_dict[cur_dir]-len(product_dict))/raw_dict[cur_dir])
                print(r"Compression rate of {0} merging last {1} digit: {2}%".format(cur_dir, str(10-digit), compress_rate))

                if digit not in compress_dict:
                    compress_dict.setdefault(digit,{})[cur_dir] = compress_rate
                else:
                    compress_dict[digit][cur_dir] = compress_rate

            ##to create merged directories
            merge_images(product_dict, cur_dir)
            print()