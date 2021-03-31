
import re, numpy as np
import torch, pandas as pd
from word2number import w2n
import re
import en_core_web_lg
nlp = en_core_web_lg.load()

import en_core_sci_md
nlp_sci_md = en_core_sci_md.load()

american_number_system = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60,
        'seventy': 70,
        'eighty': 80,
        'ninety': 90,
        'hundred': 100,
        'thousand': 1000,
        'million': 1000000,
        'billion': 1000000000,
        'point': '.'
    }

def get_token_span(tokens, text):
#     token_spans = []
    start_offsets = []
    end_offsets = []
    for j in range(len(tokens)):
        if j == 0:
            token_offset = text.find(tokens[j])
        else:
            offset_begin = last_token_offset + len(tokens[j-1])
            text_to_find = text[offset_begin:]
            additional_offset = text_to_find.find(tokens[j])
            if additional_offset == -1 or additional_offset > len(tokens[j])*2:
                additional_offset = text_to_find.find(tokens[j][0])
            token_offset = offset_begin + additional_offset
#         token_spans.append((token_offset, token_offset+len(tokens[j])))
        start_offsets.append(token_offset)
        end_offsets.append(token_offset+len(tokens[j]))
        last_token_offset = token_offset
    return start_offsets, end_offsets

def isNumber(ent):   
    try:
        w2n.word_to_num(ent.text)
    
    except Exception as e:
        return False, 
        
    number_sentence = ent.text.rstrip("s")
    number_sentence = number_sentence.replace('-', ' ')
    number_sentence = number_sentence.lower()  # converting input to lowercase

    split_words = number_sentence.strip().split()  # strip extra spaces and split sentence into words

    if split_words[-1] not in american_number_system:
        new_char = ent.end_char - len(split_words[-1]) - 1
        return True, new_char
    return True,

    

def get_ent_offsets(text):
    doc_sci = nlp_sci_md(text)
    start_offsets = [ent.start_char for ent in doc_sci.ents]
    end_offsets = [ent.end_char for ent in doc_sci.ents]
    return start_offsets, end_offsets


def get_text_num(text):
    num_positions, num_norm, num_texts = get_number(text)
    last_end = 0
    new_text = ""
    if len(num_positions) == 0:
        return text, [], []
    for num_pos in num_positions:
        new_text += text[last_end:num_pos[0]] + " [NUM] "
        last_end = num_pos[1]
    new_text += text[last_end:len(text)]
#     new_texts.append(new_text)
#     input_ids = tokenizer.encode(new_text)
#     nums = []
#     nums = [in_id for in_id in input_ids if in_id == 1]
#     if (len(nums) != len(num_positions)):
#         print(new_text, num_norm)
    return new_text, num_norm, num_texts


def get_new_answer_char(orig_pos, new_pos, char):
    index = np.digitize(char, orig_pos) - 1
    if index < 0:
        return char
    new_char = new_pos[index] - orig_pos[index] + char
    
    return new_char

def get_text_num_with_answers(text,  answers):
    num_positions, num_norm, num_texts = get_number(text)
    last_end = 0
    new_text = ""
    orig_pos = []
    new_pos = []
    
   
    for num_pos in num_positions:
        new_text += text[last_end:num_pos[0]] + " [NUM] "
        last_end = num_pos[1]
        
        if len(orig_pos) == 0:
            new_pos.append(num_pos[0] + 7)
        else:
            new_pos.append(new_pos[-1] + num_pos[0] - orig_pos[-1] + 7 )
        orig_pos.append(num_pos[1])
    
   
    new_text += text[last_end:len(text)]
   
    new_answers = []
    for answer in answers:
        answer_start = answer['answer_start']
        answer_text = answer['text']
        new_answer_start = get_new_answer_char(orig_pos, new_pos, answer_start)
        new_answer_end = get_new_answer_char(orig_pos, new_pos, answer_start+len(answer_text))
        new_answer_text = new_text[new_answer_start: new_answer_end]
        new_answers.append({'text':new_answer_text, 'answer_start':new_answer_start})
#     new_texts.append(new_text)

    return new_text, num_norm, new_answers,num_texts

def get_text_num_with_answer(text, answer_start, answer_text):
    num_positions, num_norm, num_texts = get_number(text)
    last_end = 0
    new_text = ""
    orig_pos = []
    new_pos = []
    
   
    for num_pos in num_positions:
        new_text += text[last_end:num_pos[0]] + " [NUM] "
        last_end = num_pos[1]
        
        if len(orig_pos) == 0:
            new_pos.append(num_pos[0] + 7)
        else:
            new_pos.append(new_pos[-1] + num_pos[0] - orig_pos[-1] + 7 )
        orig_pos.append(num_pos[1])
    
   
    new_text += text[last_end:len(text)]
    if answer_text is None:
        return new_text, " ".join(num_norm), None, None
    
    new_answer_start = get_new_answer_char(orig_pos, new_pos, answer_start)
    new_answer_end = get_new_answer_char(orig_pos, new_pos, answer_start+len(answer_text))
    new_answer_text = new_text[new_answer_start: new_answer_end]
#     new_texts.append(new_text)

    return new_text, num_norm, new_answer_start, new_answer_text, num_texts
from dateutil.parser import parse

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except Exception as e:
        return False

def get_date_offsets(ents):
    starts = []
    ends = []
    for ent in ents:
        if ent.label_ == "DATE" and is_date(ent.text):
            starts.append(ent.start_char)
            ends.append(ent.end_char)
    return starts, ends

def get_number(text):
    num_positions = []
    num_norm = []
    num_texts = []
    matches = re.finditer("\d[\d,.]*", text)
    
    ent_starts, ent_ends = get_ent_offsets(text)
#     date_starts, date_ends = get_date_offsets(ents)

    doc = nlp(text)
    ents= doc.ents
    i = 0
    skip = False
    for match in matches:
        nums = []
        
        s = match.start()
        e = match.end()
        if text[s:e].endswith('.') or text[s:e].endswith(','):
            e -= 1
        while (i < len(ents) and s > ents[i].start_char ):
            
            try:
                if ents[i].label_ in ["CARDINAL", "QUANTITY", "PERCENT"] and ents[i].end_char < s: 
                    isnumber = isNumber(ents[i])
                    if isnumber[0]:
                        if len(isnumber) == 1:
                            num_text = ents[i].text.replace(",", "")
                            num_norm.append(w2n.word_to_num(num_text))
                            num_positions.append([ents[i].start_char, ents[i].end_char])
                            
                         
                        else:
#                             print(ents[i].text, ents[i].label_)
                            num_norm.append(w2n.word_to_num(text[ents[i].start_char:isnumber[1]]))
                            num_positions.append([ents[i].start_char, isnumber[1]])
                            
                    
                i+=1
            except Exception as exception: 
                print(exception)
                print(ents[i].text, " ", ents[i].label_)
                i+=1
                continue
#             if (i < len(ents) and s > ents[i].start_char
#                 and e <= ents[i].end_char and ents[i].label_ == "PRODUCT"):
        if i < len(ents) and s == ents[i].start_char:
            i+=1
        if s > 0 and "." not in text[s:e] and text[s-1] == ".":
            s -= 1
        num_text = text[s:e].replace(",", "")
        
        
            
        try:
            number = float(num_text)
        except Exception as exception: 
            print(exception)
            continue
            
        if s > 0 and text[s-1].isalpha():
            continue
            
        if (len(ent_starts) > 0):
            ent_index = np.digitize(s, ent_starts) -1
            ent_text = text[ent_starts[ent_index]:ent_ends[ent_index]]
            
            if (e <= ent_ends[ent_index] and "=" not in ent_text and "<" not in ent_text):
                continue
        
        num_positions.append([s, e])
        num_norm.append(number)
    num_texts = [text[p[0]:p[1]] for p in num_positions]
    
    return num_positions, num_norm, num_texts

