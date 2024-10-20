import faiss
import pickle
import numpy as np

from env.consts import CHOSEN_EMB_FOL, CHOSEN_PARAMS, SHOT_INFO_POSITION

def retrieve_shots_from_query(chosen_avatar_emb_path: str, faiss_index_emb_path: str, read_files_emb_path: str, topK: int) -> set():
    global faiss
    set_result = set()
    index = faiss.read_index(faiss_index_emb_path)
    with open(chosen_avatar_emb_path, 'rb') as f:
        chosen_character_emb = np.array(pickle.load(f))
    with open(read_files_emb_path, 'rb') as f:
        read = pickle.load(f)

    _, indices = index.search(chosen_character_emb, k=topK)

    for i in indices[0]:
        face_relevant_emb_path = read[i]
        face_relevant_img_path = face_relevant_emb_path.replace("./", "/").replace("faces_emb", "faces").replace("pkl", "jpg").replace(f"-emb_{CHOSEN_PARAMS[CHOSEN_EMB_FOL]}", "")
        shot_info = extract_shot_from_face_img(face_relevant_img_path)
        set_result.add(shot_info)
    return set_result

def get_movie(shot_i):
    arr_str = shot_i.split('-')
    selected_movie = arr_str[0]
    return selected_movie

def get_set_of_result_shots(result: dict) -> set:
    result_lst = list()
    for _, value in result.items():
        result_lst.extend(value)
    return set(result_lst)

def extract_shot_from_face_img(face_img: str) -> str:
    return face_img.split('/')[SHOT_INFO_POSITION]
