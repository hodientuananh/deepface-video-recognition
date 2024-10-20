from env.consts import CHOSEN_CHARACTER, CHOSEN_EMB_FOL, CHOSEN_FACE_DET, CHOSEN_MOVIE, CHOSEN_MOVIE_FOL, \
    CHOSEN_PARAMS, ROOT_EVALUATION, ROOT_FAISS_INDEX, ROOT_FEATURES_QUERY

def get_face_det_model_emb_name():
    return f'det_{CHOSEN_PARAMS[CHOSEN_FACE_DET]}-emb_{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}'

def get_chosen_avatar_dir_path():
    return f"{ROOT_FEATURES_QUERY}/{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}/{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}/{CHOSEN_PARAMS[CHOSEN_CHARACTER]}/"

def get_faiss_index_path():
    return f"{ROOT_FAISS_INDEX}/{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}/{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}/movie_{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}-det_{CHOSEN_PARAMS[CHOSEN_FACE_DET]}-emb_{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}.pkl"

def get_read_file_emb_path():
    return f"{ROOT_FAISS_INDEX}/{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}/{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}/read-movie_{CHOSEN_PARAMS[CHOSEN_MOVIE_FOL]}-det_{CHOSEN_PARAMS[CHOSEN_FACE_DET]}-emb_{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}.pkl"

def get_evaluation_file(topK, chosen_movie_fol, chosen_character):
    return f"{ROOT_EVALUATION}/evaluation_top{topK}_{chosen_movie_fol}_{chosen_character}.csv"

def get_chosen_avatar_emb_path(chosen_avatar_dir_path, lst_images, character_image_index):
    return f"{chosen_avatar_dir_path}/{lst_images[character_image_index].split('.')[0]}.pkl"

def get_check_key_and_character_key(character_image_index):
    chk_key = f'chk_{character_image_index}'
    character_key = f'character_{character_image_index}'
    return chk_key, character_key

def add_to_shot_result(shot_result, face_det_model_emb_name):
    if face_det_model_emb_name not in shot_result:
        shot_result[face_det_model_emb_name] = dict()
        
def init_chosen_movie(chosen_movie):
    CHOSEN_PARAMS[CHOSEN_MOVIE] = chosen_movie

def init_chosen_movie_fol(chosen_movie_fol):
    CHOSEN_PARAMS[CHOSEN_MOVIE_FOL] = chosen_movie_fol
    
def init_chosen_character(chosen_character):
    CHOSEN_PARAMS[CHOSEN_CHARACTER] = chosen_character
    
def init_chosen_face_det(chosen_face_det):
    CHOSEN_PARAMS[CHOSEN_FACE_DET] = chosen_face_det
    
def init_chosen_emb_fol(chosen_emb_fol):
    CHOSEN_PARAMS[CHOSEN_EMB_FOL] = chosen_emb_fol
