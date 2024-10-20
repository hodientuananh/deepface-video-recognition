MOVIES_DIR_MAPPING = {
    'Memphis': 'Memphis',
    'Calloused Hands': 'Calloused_Hands',
    'Liberty Kid': 'Liberty_Kid',
    'Losing Ground': 'losing_ground',
    'Like Me': 'like_me'
}
MOVIES_CHARACTERS_MAPPING = {
    'Memphis': ['willis'],
    'Calloused Hands': ['Byrd', 'Debbie'],
    'Liberty Kid': ['Derrick'],
    'Losing Ground': ['sara'],
    'Like Me': ['Burt', 'Kiya']
}
EMB_MODELS = {
    "ArcFace": "ArcFace",
    "VGG-Face": "VggFace",
    "Facenet": "FaceNet",
    "GhostFaceNet": "GhostFaceNet"
}
DET_MODELS = ['fastmtcnn', 'retinaface', 'opencv']
DET_EMB_MAPPING = (
    ('fastmtcnn', 'ArcFace'),
    ('fastmtcnn', 'VggFace'),
    ('fastmtcnn', 'FaceNet'),
    ('fastmtcnn', 'GhostFaceNet'),
    ('retinaface', 'ArcFace'),
    ('retinaface', 'VggFace'),
    ('retinaface', 'FaceNet'),
    ('retinaface', 'GhostFaceNet'),
    ('opencv', 'ArcFace'),
    ('opencv', 'VggFace'),
    ('opencv', 'FaceNet'),
    ('opencv', 'GhostFaceNet')
)

CHOSEN_MOVIE = 'chosen_movie'
CHOSEN_MOVIE_FOL = 'chosen_movie_fol'
CHOSEN_CHARACTER = 'chosen_character'
CHOSEN_FACE_DET = 'chosen_face_det'
CHOSEN_EMB_MODEL = 'chosen_emb_model'
CHOSEN_EMB_FOL = 'chosen_emb_fol'
CHOSEN_PARAMS = {
    CHOSEN_MOVIE: None,
    CHOSEN_MOVIE_FOL: None,
    CHOSEN_CHARACTER: None,
    CHOSEN_FACE_DET: None,
    CHOSEN_EMB_MODEL: None,
    CHOSEN_EMB_FOL: None,
}

## Directory paths
ROOT_GROUND_TRUTH ="data/ground_truth"
ROOT_QUERY ="data/character_query"
ROOT_SHOTS ="data/shots"
ROOT_THUMBNAIL ="data/thumbnail"
ROOT_FEATURES_QUERY = "data/character_emb_query"
ROOT_FAISS_INDEX = "data/faiss_index"
ROOT_EVALUATION = "evaluation"

BATCH_SIZE = 10
SHOT_INFO_POSITION = -2

ROW_SIZE_MIN = 1
ROW_SIZE_MAX = 6
ROW_SIZE_INIT = 3

## Top K
TOP_K = 5
TOP_K_MIN = 1
TOP_K_MAX = 1000
TOP_K_STEP = 1
TOP_K_INIT = 5
