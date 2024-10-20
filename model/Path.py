from env.consts import CHOSEN_CHARACTER, CHOSEN_EMB_FOL, CHOSEN_FACE_DET, CHOSEN_MOVIE_FOL, CHOSEN_PARAMS, ROOT_EVALUATION, ROOT_FAISS_INDEX, ROOT_FEATURES_QUERY

class Path:
    face_det_model_emb_name = None
    
    def __init__(self) -> None:
        pass
    
    def get_face_det_model_emb_name(self):
        return self.face_det_model_emb_name
    
    def set_face_det_model_emb_name(self, face_det, emb_fol):
        self.face_det_model_emb_name = f'det_{face_det}-emb_{emb_fol}'
    
    def get_chosen_avatar_dir_path(self):
        return f"{ROOT_FEATURES_QUERY}/{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}/{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}/{CHOSEN_PARAMS[CHOSEN_CHARACTER]}/"
    
    def get_faiss_index_path(self):
        return f"{ROOT_FAISS_INDEX}/{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}/{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}/movie_{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}-det_{CHOSEN_PARAMS[CHOSEN_FACE_DET]}-emb_{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}.pkl"
    
    def get_read_file_emb_path(self):
        return f"{ROOT_FAISS_INDEX}/{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}/{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}/read-movie_{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}-det_{CHOSEN_PARAMS[CHOSEN_FACE_DET]}-emb_{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}.pkl"
    
    def get_evaluation_file(self, topK, chosen_movie_fol, chosen_character):
        return f"{ROOT_EVALUATION}/evaluation_top{topK}_{chosen_movie_fol}_{chosen_character}.csv"
    
    def get_chosen_avatar_emb_path(self, chosen_avatar_dir_path, lst_images, character_image_index):
        return f"{chosen_avatar_dir_path}/{lst_images[character_image_index].split('.')[0]}.pkl"
    
    def get_check_key_and_character_key(self, character_image_index):
        chk_key = f'chk_{character_image_index}'
        character_key = f'character_{character_image_index}'
        return chk_key, character_key
    
    
    