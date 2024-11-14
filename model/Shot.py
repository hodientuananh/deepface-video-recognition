from model.Path import Path
from env.consts import DEFAULT_TOP_K_FRAMES, SHOT_INFO_POSITION

import faiss
import pickle
import numpy as np

class Shot:
    result = dict()
    path = Path()
    chk_key_lst = []
    weight_dict = dict()
    average_distance_dict = dict()
    
    def __init__(self, path: Path) -> None:
        self.path = path
        
    def add_shot_to_weight_dict(self, shot_info: str, distance: float):
        if shot_info not in self.weight_dict:
            self.weight_dict[shot_info] = []
        self.weight_dict[shot_info].append(distance)
    
    def cal_average_distance(self, distance_lst: list) -> float:
        return sum(distance_lst) / len(distance_lst)
    
    def cal_weight_distance(self):
        for shot_info, distance_lst in self.weight_dict.items():
            self.average_distance_dict[shot_info] = self.cal_average_distance(distance_lst)
        
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
        distances, indices = index.search(chosen_character_emb, k=topK)
        return distances[0], indices[0]
    
    def get_face_relevant_img_path(self, face_relevant_emb_path: str):
        return face_relevant_emb_path.replace("./", "/")\
            .replace("faces_emb", "faces").replace("pkl", "jpg")\
                .replace(f"-emb_{self.path.get_chosen_emb_fol()}", "")
                
    def get_weight_dict_from_topK_frames(self) -> dict:
        self.weight_dict = dict()
        index = self.load_index()
        chosen_character_emb = self.load_chosen_character_emb()
        distances, indices = self.search_topK_frames(index, chosen_character_emb, DEFAULT_TOP_K_FRAMES)
        read = self.load_read_file_emb()
        for i in range(len(indices)):
            face_relevant_emb_path = read[indices[i]]
            face_relevant_dist = distances[i]
            face_relevant_img_path = self.get_face_relevant_img_path(face_relevant_emb_path)
            shot_info = self.extract_shot_from_face_img(face_relevant_img_path)
            self.add_shot_to_weight_dict(shot_info, face_relevant_dist)
            self.cal_weight_distance()
        return self.average_distance_dict
    
    # def sorted_dict_by_weight(self, dict_result: dict) -> list:
    #     return sorted(dict_result.items(), key=lambda item: item[1])
    
    def cal_sorted_weight_per_shot(self, weight_per_shot: dict) -> dict:
        return dict(sorted(weight_per_shot.items(), key=lambda item: item[1]))
         
    def get_topK_shots(self, weight_per_shot: dict) -> dict:
        sorted_shots = self.cal_sorted_weight_per_shot(weight_per_shot)
        return dict(list(sorted_shots.items())[:self.path.get_topK()])
    
    def get_list_distance_per_shot(self) -> dict:
        weight_lst_per_shot = {}
        for _, dict_shot_character in self.result[self.path.get_face_det_model_emb_name()].items():
            for shot, distance in dict_shot_character.items():
                if shot not in weight_lst_per_shot:
                    weight_lst_per_shot[shot] = [distance]
                else:
                    weight_lst_per_shot[shot].append(distance)
        return weight_lst_per_shot

    def cal_average_distance_per_shot(self, weight_lst_per_shot: dict) -> dict:
        weight_per_shot = {}
        for shot, distance_lst in weight_lst_per_shot.items():
            weight_per_shot[shot] = sum(distance_lst)/len(distance_lst)
        return weight_per_shot
    
    def get_shots_per_character(self) -> dict:
        dict_result = self.get_weight_dict_from_topK_frames()
        return self.get_topK_shots(dict_result)

    def get_sorted_weight_dict_per_character(self) -> dict:
        # dict_predict = dict()
        # for character_key, _ in self.result[self.path.get_face_det_model_emb_name()].items():
        #     character_predict_lst = self.result[self.path.get_face_det_model_emb_name()][character_key]
        #     dict_predict = merge_2_lst_to_frequency_dict(character_predict_lst, [], dict_predict)
        # return dict_predict
        weight_lst_per_shot = self.get_list_distance_per_shot()
        weight_per_shot = self.cal_average_distance_per_shot(weight_lst_per_shot)
        sorted_weight_dict_per_character = self.cal_sorted_weight_per_shot(weight_per_shot)
        return sorted_weight_dict_per_character

    # def get_set_of_result_shots(self, result: dict) -> set:
    #     result_lst = list()
    #     for _, value in result.items():
    #         result_lst.extend(value)
    #     return set(result_lst)
    
    def init_shot_result(self) -> None:
        face_det_model_emb_name = self.path.get_face_det_model_emb_name()
        if face_det_model_emb_name in self.result and len(face_det_model_emb_name) > 0:
            self.result[self.path.get_face_det_model_emb_name()].clear()
        self.result[self.path.get_face_det_model_emb_name()] = dict()     
            
    def add_to_shot_result(self, character_key: str, character_result_dict: dict) -> None:
        self.result[self.path.get_face_det_model_emb_name()][character_key] = dict()
        self.result[self.path.get_face_det_model_emb_name()][character_key].update(character_result_dict) 
            
    def extract_shot_from_face_img(self, face_img: str) -> str:
        return face_img.split('/')[SHOT_INFO_POSITION]      
    
    def get_shot_result(self) -> dict:
        return self.result
    
    def get_path(self) -> Path:
        return self.path
    
    def set_path(self, path: Path):
        self.path = path
        
    def set_chk_key_lst(self, chk_key_lst: list):
        self.chk_key_lst = chk_key_lst
        
    def get_chk_key_lst(self) -> list:
        return self.chk_key_lst