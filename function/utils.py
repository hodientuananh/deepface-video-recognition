import numpy as np

def convert_set_to_list(set_: set) -> list:
    return list(set_)

def convert_list_to_numpy_array(lst: list):
    return np.array(lst)

def add_element_to_dict(dict, element):
    if element in dict:
        dict[element] += 1
    else:
        dict[element] = 1
        
def add_list_to_dict(dict, lst):
    for element in lst:
        add_element_to_dict(dict, element)
        
def sort_most_frequency_dict_value_basing(frequency_dict: dict) -> dict:
    return dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))

def convert_dict_to_list(dict):
    return list(dict.keys())

def merge_2_lst_to_frequency_dict(list_1, list_2, dict) -> dict:
    frequency_dict = dict.copy()
    while len(list_1) > 0:
        while len(list_2) > 0:
            first_element = list_1[0]
            second_element = list_2[0]
            
            if first_element > second_element:
                add_element_to_dict(frequency_dict, second_element)
                list_2.remove(second_element)
            elif first_element < second_element:
                add_element_to_dict(frequency_dict, first_element)
                list_1.remove(first_element)
            else:
                add_element_to_dict(frequency_dict, first_element)
                add_element_to_dict(frequency_dict, second_element)
                list_1.remove(first_element)
                list_2.remove(second_element)   
            break
        break
        
    if len(list_1) > 0:
        add_list_to_dict(frequency_dict, list_1)
    if len(list_2) > 0:
        add_list_to_dict(frequency_dict, list_2)
    return frequency_dict

def get_topK_most_frequent_elements(dict_frequency, K) -> list:
    dict_frequency = sort_most_frequency_dict_value_basing(dict_frequency)
    return convert_dict_to_list(dict_frequency)[:K]

def get_check_key_and_character_key(character_image_index):
    chk_key = f'chk_{character_image_index}'
    character_key = f'character_{character_image_index}'
    return chk_key, character_key

def get_movie(shot_i: int) -> str:
        arr_str = shot_i.split('-')
        selected_movie = arr_str[0]
        return selected_movie