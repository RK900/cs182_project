import numpy as np
import re
import num2words
from unidecode import unidecode

def preprocess(data):
    out = []
    for review in data:
        #Replace numbers with word repr
        text = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), review) 
        #Remove all extra whitespaces by splitting string
        text_list = text.split()
        #Dedup text 
        deduped_text = [word for i, word in enumerate(text_list) if i == 0 or word != text_list[i-1]]
        text = " ".join(deduped_text) 
        #Replace unicode chars with ascii
        text = unidecode(text)
        out.append(text)
        
    return out
