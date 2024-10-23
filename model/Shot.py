from function.utils import convert_set_to_list, merge_2_lst_to_frequency_dict
from model.Path import Path
from env.consts import SHOT_INFO_POSITION

import faiss
import pickle
import numpy as np

class Shot:
    result = dict()
    path = Path()
    
    def __init__(self, path: Path) -> None:
        self.path = path
    
    def get_shots_from_query(self) -> set():
        global faiss
        set_result = set()
        index = faiss.read_index(self.path.get_faiss_index_path())
        with open(self.path.get_chosen_avatar_emb_path(), 'rb') as f:
            chosen_character_emb = np.array(pickle.load(f))
        with open(self.path.get_read_file_emb_path(), 'rb') as f:
            read = pickle.load(f)

        _, indices = index.search(chosen_character_emb, k=self.path.get_topK())

        for i in indices[0]:
            face_relevant_emb_path = read[i]
            face_relevant_img_path = face_relevant_emb_path.replace("./", "/")\
                .replace("faces_emb", "faces").replace("pkl", "jpg")\
                    .replace(f"-emb_{self.path.get_chosen_emb_fol()}", "")
            shot_info = self.extract_shot_from_face_img(face_relevant_img_path)
            set_result.add(shot_info)
        return set_result

    def get_frequency_dict_based_character(self, face_det_model_emb_name):
        dict_predict = dict()
        for character_key, _ in self.result[face_det_model_emb_name].items():
            character_predict_lst = convert_set_to_list(self.result[face_det_model_emb_name][character_key])
            dict_predict = merge_2_lst_to_frequency_dict(character_predict_lst, [], dict_predict)
        return dict_predict

    def get_set_of_result_shots(self, result: dict) -> set:
        result_lst = list()
        for _, value in result.items():
            result_lst.extend(value)
        return set(result_lst)
    
    def init_shot_result(self, face_det_model_emb_name: str) -> None:
        if face_det_model_emb_name not in self.result:
            self.result[face_det_model_emb_name] = dict()
            
    def add_to_shot_result(self, face_det_model_emb_name, character_key: str, character_result_set: set) -> None:
        self.result[face_det_model_emb_name][character_key] = set()
        self.result[face_det_model_emb_name][character_key].update(character_result_set)  
            
    def extract_shot_from_face_img(self, face_img: str) -> str:
        return face_img.split('/')[SHOT_INFO_POSITION]      
    
    def get_shot_result(self):
        return self.result
    
    def get_path(self):
        return self.path
    
    def set_path(self, path: Path):
        self.path = path