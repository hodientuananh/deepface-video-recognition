# from model.Path import Path
# from model.Shot import Shot
# from env.consts import CHOSEN_PARAMS, EMBBEDDING, SHOT_INFO_POSITION

# import faiss
# import pickle
# import numpy as np

# class Extraction:
#     path = Path()
#     shot = Shot()
#     topK = None
    
#     def __init__(self, path: Path, shot: Shot) -> None:
#         self.path = path
#         self.shot = shot

#     def get_shots_from_query(self) -> set():
#         global faiss
#         set_result = set()
#         index = faiss.read_index(self.path.get_faiss_index_path())
#         with open(self.path.get_chosen_avatar_emb_path(), 'rb') as f:
#             chosen_character_emb = np.array(pickle.load(f))
#         with open(self.path.get_read_file_emb_path(), 'rb') as f:
#             read = pickle.load(f)

#         _, indices = index.search(chosen_character_emb, k=self.path.get_topK())

#         for i in indices[0]:
#             face_relevant_emb_path = read[i]
#             face_relevant_img_path = face_relevant_emb_path.replace("./", "/").replace("faces_emb", "faces").replace("pkl", "jpg").replace(f"-emb_{CHOSEN_PARAMS[EMBBEDDING.FOLDER]}", "")
#             shot_info = self.shot.extract_shot_from_face_img(face_relevant_img_path)
#             set_result.add(shot_info)
#         return set_result

#     def get_movie(self, shot_i: int) -> str:
#         arr_str = shot_i.split('-')
#         selected_movie = arr_str[0]
#         return selected_movie

#     def get_set_of_result_shots(self, result: dict) -> set:
#         result_lst = list()
#         for _, value in result.items():
#             result_lst.extend(value)
#         return set(result_lst)