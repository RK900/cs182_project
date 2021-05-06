import numpy as np
from data_parsing import load_dataset
import re
import num2words
import json
from unidecode import unidecode

def preprocess(in_file, out_file):
    data = load_dataset(in_file)
    out = []
    for d in data:
        review, stars = d[0], d[1]
        #Replace numbers with word repr
        text = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), review) 
        #Remove all extra whitespaces by splitting string
        text_list = text.split()
        #Dedup text 
        deduped_text = [word for i, word in enumerate(text_list) if i == 0 or word != text_list[i-1]]
        text = " ".join(deduped_text) 
        #Replace unicode chars with ascii
        text = unidecode(text)
        dict_obj = {}
        dict_obj['text'], dict_obj['stars'] = text, stars
        out.append(dict_obj)
        
    with open(out_file, "w") as write_file:
        json.dump(out, write_file, indent=4)
    write_file.close()