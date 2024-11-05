from function.utils import convert_set_to_list, merge_2_lst_to_frequency_dict
from model.Path import Path
from env.consts import DEFAULT_TOP_K_FRAMES, SHOT_INFO_POSITION

import faiss
import pickle
import numpy as np

class Shot:
    result = dict()
    path = Path()
    
    def __init__(self, path: Path) -> None:
        self.path = path
        
    def load_index(self):
        global faiss
        return faiss.read_index(self.path.get_faiss_index_path())
    
    def load_chosen_character_emb(self):
        with open(self.path.get_chosen_avatar_emb_path(), 'rb') as f:
            return np.array(pickle.load(f))
        
    def load_read_file_emb(self):
        with open(self.path.get_read_file_emb_path(), 'rb') as f:
            return pickle.load(f)
        
    def search_topK_frames(self, index, chosen_character_emb, topK):
        _, indices = index.search(chosen_character_emb, k=topK)
        return indices[0]
    
    def get_face_relevant_img_path(self, face_relevant_emb_path: str):
        return face_relevant_emb_path.replace("./", "/")\
            .replace("faces_emb", "faces").replace("pkl", "jpg")\
                .replace(f"-emb_{self.path.get_chosen_emb_fol()}", "")
                
    def get_set_from_topK_frames(self) -> set:
        set_result = set()
        index = self.load_index()
        chosen_character_emb = self.load_chosen_character_emb()
        indices = self.search_topK_frames(index, chosen_character_emb, DEFAULT_TOP_K_FRAMES)
        read = self.load_read_file_emb()
        for i in indices:
            face_relevant_emb_path = read[i]
            face_relevant_img_path = self.get_face_relevant_img_path(face_relevant_emb_path)
            shot_info = self.extract_shot_from_face_img(face_relevant_img_path)
            set_result.add(shot_info)
        return set_result
                
    def get_topK_shots(self, set_result: set) -> list:
        return list(set_result)[:self.path.get_topK()]
    
    def get_shots_per_character(self) -> list:
        set_result = self.get_set_from_topK_frames()
        return self.get_topK_shots(set_result)

    def get_frequency_dict_based_character(self) -> dict:
        dict_predict = dict()
        for character_key, _ in self.result[self.path.get_face_det_model_emb_name()].items():
            character_predict_lst = convert_set_to_list(self.result[self.path.get_face_det_model_emb_name()][character_key])
            dict_predict = merge_2_lst_to_frequency_dict(character_predict_lst, [], dict_predict)
        return dict_predict

    def get_set_of_result_shots(self, result: dict) -> set:
        result_lst = list()
        for _, value in result.items():
            result_lst.extend(value)
        return set(result_lst)
    
    def init_shot_result(self) -> None:
        if self.path.get_face_det_model_emb_name() not in self.result:
            self.result[self.path.get_face_det_model_emb_name()] = dict()
            
    def add_to_shot_result(self, character_key: str, character_result_lst: list) -> None:
        self.result[self.path.get_face_det_model_emb_name()][character_key] = list()
        self.result[self.path.get_face_det_model_emb_name()][character_key].extend(character_result_lst) 
            
    def extract_shot_from_face_img(self, face_img: str) -> str:
        return face_img.split('/')[SHOT_INFO_POSITION]      
    
    def get_shot_result(self):
        return self.result
    
    def get_path(self):
        return self.path
    
    def set_path(self, path: Path):
        self.path = path