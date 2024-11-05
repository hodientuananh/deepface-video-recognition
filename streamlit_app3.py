import os
import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image
from math import ceil

from function.utils import convert_list_to_numpy_array, get_topK_most_frequent_elements, \
    get_check_key_and_character_key, get_movie
from env.consts import CHOSEN_PARAMS, DET_EMB_MAPPING, DET_MODELS, DETECTION, EMB_MODELS, EMBBEDDING, MOVIE, MOVIES_CHARACTERS_MAPPING, \
    ROOT_SHOTS, MOVIES_DIR_MAPPING, ROOT_GROUND_TRUTH, ROOT_QUERY, ROOT_SHOTS, BATCH_SIZE, \
    ROOT_THUMBNAIL, ROW_SIZE_MIN, ROW_SIZE_MAX, ROW_SIZE_INIT, TOP_K_MIN, TOP_K_MAX, TOP_K_STEP, TOP_K_INIT
    
from model.Evaluation import Evaluation
from model.Path import Path
from model.Shot import Shot

# ###################################################################################################
# GENERAL SETTINGS
## Global variables
chosen_emb_model = None
chosen_face_det = None
chosen_emb_fol = None
shot_result = dict()
lst_groundtruth = list()
lst_predict = list()
set_predict = set()
dict_predict = dict()

Images = []

chart_data = pd.DataFrame()
ap = 0

columns = np.array([])
rows = np.array([])

path = Path()
# ###################################################################################################
# INIT STREAMLIT APP
## Parameters
st.sidebar.markdown("<h1 style='text-align:left; font-size: 18px;'>Select Params</h1>",unsafe_allow_html=True)
ROW_SIZE = st.sidebar.select_slider("Row size:", range(ROW_SIZE_MIN, ROW_SIZE_MAX), value = ROW_SIZE_INIT)
topK = st.sidebar.number_input('Choose top_k result', min_value=TOP_K_MIN, max_value=TOP_K_MAX, step=TOP_K_STEP, value=TOP_K_INIT)

st.header("Movie Character Retrieval System")
st.subheader("Query Image")

## Input Image Query
st.sidebar.markdown("<h1 style='text-align:left; font-size: 18px;'>Select Input Image Query</h1>",unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='font-size: 16px;'>Choose movie</h1>", unsafe_allow_html=True)
# chosen_movie = st.sidebar.selectbox('Choose movie', list(MOVIES_DIR_MAPPING.keys()), label_visibility="collapsed")
CHOSEN_PARAMS[MOVIE.NAME] = st.sidebar.selectbox('Choose movie', list(MOVIES_DIR_MAPPING.keys()), label_visibility="collapsed")    

st.sidebar.markdown("<h1 style='font-size: 16px;'>Choose character</h1>", unsafe_allow_html=True)
option_characters = MOVIES_CHARACTERS_MAPPING[CHOSEN_PARAMS[MOVIE.NAME]]
CHOSEN_PARAMS[MOVIE.FOLDER] = MOVIES_DIR_MAPPING[CHOSEN_PARAMS[MOVIE.NAME]]
CHOSEN_PARAMS[MOVIE.CHARACTER] = st.sidebar.selectbox('Choose character', option_characters , label_visibility="collapsed")

st.sidebar.markdown("<h1 style='font-size: 16px;'>Want to compare between Face Detection - Model Embedding</h1>", unsafe_allow_html=True)
compare_bw_face_and_em = st.sidebar.checkbox("Yes", value=False)

if not compare_bw_face_and_em:
    st.sidebar.markdown("<h1 style='font-size: 16px;'>Choose face detection type</h1>", unsafe_allow_html=True)
    CHOSEN_PARAMS[DETECTION.MODEL] = st.sidebar.selectbox('Choose model detection type', DET_MODELS, label_visibility="collapsed")

    st.sidebar.markdown("<h1 style='font-size: 16px;'>Choose model embedding type</h1>", unsafe_allow_html=True)
    CHOSEN_PARAMS[EMBBEDDING.MODEL] = st.sidebar.selectbox('Choose model embedding type', list(EMB_MODELS.keys()), label_visibility="collapsed")
    CHOSEN_PARAMS[EMBBEDDING.FOLDER] = EMB_MODELS[CHOSEN_PARAMS[EMBBEDDING.MODEL]]

path_to_imgs_query = os.path.join(ROOT_QUERY, CHOSEN_PARAMS[MOVIE.FOLDER], CHOSEN_PARAMS[MOVIE.CHARACTER])
lst_images = sorted(os.listdir(path_to_imgs_query))

## Initialize session state for checkboxes if not already done
if 'checkbox_states' not in st.session_state:
    st.session_state.checkbox_states = {f'chk_{i}': False for i in range(len(lst_images))}

columns_imgs = st.columns(len(lst_images))
for file in lst_images:
    image = Image.open(os.path.join(path_to_imgs_query, file))
    image = image.resize((120,120))
    Images.append(image)
for idx, Image in enumerate(Images):
    str_key = "chk_{}".format(idx)
    with columns_imgs[idx]:
        st.image(Image)
        st.checkbox(lst_images[idx].split('.')[0], key=str_key, value=True)
        
# ###################################################################################################
# INIT GLOBAL PARAMS
## Initialize global parameters
path.set_chosen_movie(CHOSEN_PARAMS[MOVIE.NAME])
path.set_chosen_movie_fol(CHOSEN_PARAMS[MOVIE.FOLDER])
path.set_chosen_character(CHOSEN_PARAMS[MOVIE.CHARACTER])

path.set_topK(topK)
path.set_evaluation_file(topK)

shot = Shot(path)
# ###################################################################################################
# RETRIEVE THE TOP K RESULTS
## Show result of specific detection and embedding model
if compare_bw_face_and_em == False:
    for character_image_index in range(len(lst_images)):
        chk_key, character_key = get_check_key_and_character_key(character_image_index)
        path.set_global_path(CHOSEN_PARAMS[DETECTION.MODEL], CHOSEN_PARAMS[EMBBEDDING.FOLDER])
        shot.init_shot_result()
        if chk_key in st.session_state and st.session_state[chk_key]:
            path.set_chosen_avatar_emb_path(character_image_index, lst_images)
            character_result_lst = shot.get_shots_per_character()
            shot.add_to_shot_result(character_key, character_result_lst)
## Show result of all detection and embedding model
else:
    for character_image_index in range(len(lst_images)):
        chk_key, character_key = get_check_key_and_character_key(character_image_index)   
        for (face_det, emb_fol) in DET_EMB_MAPPING:
            path.set_global_path(face_det, emb_fol)
            shot.init_shot_result()
            if chk_key in st.session_state and st.session_state[chk_key]:
                path.set_chosen_avatar_emb_path(character_image_index, lst_images)
                character_result_set = shot.get_shots_per_character()
                shot.add_to_shot_result(character_key, character_result_set)

# # ######################################################################################################
# EVALUATION METRICS
## Ground truth
path_to_ground_truth = os.path.join(ROOT_GROUND_TRUTH, CHOSEN_PARAMS[MOVIE.FOLDER],"{}.xlsx".format(CHOSEN_PARAMS[MOVIE.CHARACTER]))
df = pd.read_excel(path_to_ground_truth)
df_ground_truth_full = df["Full"]
lst_groundtruth = [i for i in df_ground_truth_full]
evaluation_file = path.get_evaluation_file()

if not compare_bw_face_and_em:
    # face_det_model_emb_name = get_face_det_model_emb_name()
    dict_predict = shot.get_frequency_dict_based_character()
    st.write("dict_predict", dict_predict)
    lst_predict = get_topK_most_frequent_elements(dict_predict, topK)
    st.write("lst_predict", lst_predict)
    evaluation = Evaluation(lst_predict, lst_groundtruth)
    precision_scores, recall_scores, f1_score, ap = evaluation.calculate_metrics()
    st.write('sum(precision_scores)', sum(precision_scores), 'len(lst_groundtruth)', len(lst_groundtruth))
    chart_data = pd.DataFrame(
        {
            "Precsion": convert_list_to_numpy_array(precision_scores),
            "Recall": convert_list_to_numpy_array(recall_scores),
            "F1": convert_list_to_numpy_array(f1_score),
        }
    )
    st.table(chart_data)
    st.line_chart(chart_data)
    st.write("Average Precision:", ap)
else:
    for (face_det, emb_fol) in DET_EMB_MAPPING:
        path.set_global_path(face_det, emb_fol)
        dict_predict = shot.get_frequency_dict_based_character()
        lst_predict = get_topK_most_frequent_elements(dict_predict, topK)
        evaluation = Evaluation(lst_predict, lst_groundtruth)
        precision_scores, recall_scores, f1_score, ap = evaluation.calculate_metrics()
        rows = np.append(rows, ap)
        columns = np.append(columns, path.get_face_det_model_emb_name())

    chart_data = pd.DataFrame(
        {
            "Face Detection and Model Embedding": columns,
            "Average Precision": rows,
        }
    )
    st.bar_chart(
        chart_data,
        x="Face Detection and Model Embedding",
        y="Average Precision",
    )
    st.table(chart_data)
    chart_data.to_csv(evaluation_file, sep=',', encoding='utf-8', index=False, header=True)

# # ######################################################################################################
# SHOW THE VIDEO RESULTS
num_batches = ceil(len(lst_predict)/BATCH_SIZE)

if lst_predict not in st.session_state:
    st.session_state.lst_predict = lst_predict

if 'st_lst_result' not in st.session_state:
    st_lst_result = st.empty()

if 'st_video' not in st.session_state:
    st_video = st.empty()

if 'selected_shot' not in st.session_state:
    st.session_state.selected_shot=''
else:
    shot_i = st.session_state.selected_shot
    st.write(get_movie(shot_i))
    if len(shot_i)>0:
        arr_str = shot_i.split('-')
        selected_movie = arr_str[0]
        selected_scene =  f'{arr_str[0]}-{arr_str[1]}'
        path_to_video_file = os.path.join(ROOT_SHOTS, selected_movie,selected_scene,"{}.webm".format(shot_i))
        if os.path.exists(path_to_video_file):
            video_file = open(path_to_video_file, 'rb')
            video_bytes = video_file.read()
            st_video.video(video_bytes)
            st_lst_result.markdown(f"Playing video: {shot_i}")
        else:
            st_lst_result.markdown(f"Not found video: {shot_i}")
    else:
        st_video = st.empty()

str_infor = "Top <b>{}</b> results of <b>{}</b> in <b>{}</b>".format(len(lst_predict), path.get_chosen_character, path.get_chosen_movie)

str_results = """
<table style="border: 1px solid #cc9966; width: 100%;" cellspacing="0" cellpadding="10px">
<tbody>
<tr>
<td style="border-bottom-color: #FC3; border-bottom-style: solid; border-bottom-width: 1px;" \
    bgcolor="#ec6a00" height="20"><span style="padding-top: 10px; padding-bottom: 20px; \
    font-family: Arial; font-size: 14px; font-style: normal; color: white; font-weight: bold;">
""" +str_infor + """</span></td></tr>""" 
st.markdown(str_results, unsafe_allow_html=True)

page = st.selectbox("Page", range(0, num_batches + 1))

str_results = """
</tbody>
</table>
"""
st.markdown(str_results, unsafe_allow_html=True)

batch = lst_predict[(page-1)*BATCH_SIZE : page*BATCH_SIZE]

grid = st.columns(ROW_SIZE)
col_id = 0
for shot_i in batch:
    with grid[col_id]:
        arr_str = shot_i.split('-')
        seleted_movie = arr_str[0]
        seleted_scene =  f'{arr_str[0]}-{arr_str[1]}'

        path_to_thumb = os.path.join(ROOT_THUMBNAIL,seleted_movie, seleted_scene, shot_i)
        if not os.path.exists(path_to_thumb):
            path_to_file_thumb = f"{ROOT_THUMBNAIL}/default_thumbnail.png"
        else:
            lst_thumb = os.listdir(path_to_thumb)
            if len(lst_thumb)>0:
                path_to_file_thumb = os.path.join(path_to_thumb,lst_thumb[0])
            else:
                path_to_file_thumb = f"{ROOT_THUMBNAIL}/default_thumbnail.png"
        st.image(path_to_file_thumb, caption=shot_i,width=192)
        st.button("Play video", key=f'{shot_i}')
    col_id = (col_id + 1) % ROW_SIZE
