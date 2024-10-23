from env.consts import ROOT_EVALUATION, ROOT_FAISS_INDEX, ROOT_FEATURES_QUERY

class Path:
    face_det_model_emb_name = ''
    faiss_index_emb_path = ''
    read_files_emb_path = ''
    evaluation_file = ''
    
    chosen_avatar_dir_path = ''
    chosen_avatar_emb_path = ''
    chosen_movie = ''
    chosen_movie_fol = ''
    chosen_character = ''
    chosen_face_det = ''
    chosen_emb_fol = ''
    
    topK = 0
    
    def __init__(self) -> None:
        pass
    
    def set_global_path(self, face_det, emb_fol) -> None:
        self.set_chosen_face_det(face_det)
        self.set_chosen_emb_fol(emb_fol)
        self.set_face_det_model_emb_name()
        self.set_faiss_index_path()
        self.set_chosen_avatar_dir_path()
        self.set_read_file_emb_path()
    
    def get_chosen_movie(self) -> str:
        return self.chosen_movie
    
    def set_chosen_movie(self, chosen_movie) -> None:
        self.chosen_movie = chosen_movie
        
    def get_chosen_movie_fol(self) -> str:
        return self.chosen_movie_fol
    
    def set_chosen_movie_fol(self, chosen_movie_fol) -> None:
        self.chosen_movie_fol = chosen_movie_fol
        
    def get_chosen_character(self) -> str:
        return self.chosen_character
    
    def set_chosen_character(self, chosen_character) -> None:
        self.chosen_character = chosen_character
        
    def get_chosen_face_det(self) -> str:
        return self.chosen_face_det
    
    def set_chosen_face_det(self, chosen_face_det) -> None:
        self.chosen_face_det = chosen_face_det
        
    def get_chosen_emb_fol(self) -> str:
        return self.chosen_emb_fol
    
    def set_chosen_emb_fol(self, chosen_emb_fol) -> None:
        self.chosen_emb_fol = chosen_emb_fol
    
    def get_face_det_model_emb_name(self) -> str:
        return self.face_det_model_emb_name
    
    def set_face_det_model_emb_name(self) -> None:
        self.face_det_model_emb_name = f'det_{self.chosen_face_det}-emb_{self.chosen_emb_fol}'
    
    def get_chosen_avatar_dir_path(self) -> str:
        return self.chosen_avatar_dir_path
    
    def set_chosen_avatar_dir_path(self) -> None:
        self.chosen_avatar_dir_path = f"{ROOT_FEATURES_QUERY}/{self.chosen_movie_fol}/{self.chosen_emb_fol}/{self.chosen_character}"
    
    def get_faiss_index_path(self) -> str:
        return self.faiss_index_emb_path
    
    def set_faiss_index_path(self) -> None:
        self.faiss_index_emb_path = f"{ROOT_FAISS_INDEX}/{self.chosen_movie_fol}/{self.chosen_emb_fol}/movie_{self.chosen_movie_fol}-det_{self.chosen_face_det}-emb_{self.chosen_emb_fol}.pkl"
    
    def get_read_file_emb_path(self) -> str:
        return self.read_files_emb_path
    
    def set_read_file_emb_path(self) -> None:
        self.read_files_emb_path = f"{ROOT_FAISS_INDEX}/{self.chosen_movie_fol}/{self.chosen_emb_fol}/read-movie_{self.chosen_movie_fol}-det_{self.chosen_face_det}-emb_{self.chosen_emb_fol}.pkl"
    
    def set_evaluation_file(self, topK) -> None:
        self.evaluation_file = f"{ROOT_EVALUATION}/evaluation_top{topK}_{self.chosen_movie_fol}_{self.chosen_character}.csv"
        
    def get_evaluation_file(self) -> str:
        return self.evaluation_file
    
    def get_chosen_avatar_emb_path(self) -> str:
        return self.chosen_avatar_emb_path
        
    def set_chosen_avatar_emb_path(self, character_image_index, lst_images) -> None:
        self.chosen_avatar_emb_path = f"{self.chosen_avatar_dir_path}/{lst_images[character_image_index].split('.')[0]}.pkl"
    
    def get_check_key_and_character_key(self, character_image_index) -> tuple:
        chk_key = f'chk_{character_image_index}'
        character_key = f'character_{character_image_index}'
        return chk_key, character_key

    def set_topK(self, topK) -> None:
        self.topK = topK
        
    def get_topK(self) -> int:
        return self.topK
    