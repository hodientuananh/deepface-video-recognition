from enum import Enum

## Constants
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
    ('opencv', 'GhostFaceNet'),
)

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
# TOP_K = 100
TOP_K_MIN = 1
TOP_K_MAX = 1000
TOP_K_STEP = 1
TOP_K_INIT = 10
DEFAULT_TOP_K_FRAMES = 10000

## Chosen Params
CHOSEN_PARAMS = {}

class MOVIE(Enum):
    NAME = "Name"
    FOLDER = "Folder"
    CHARACTER = "Character"
    
class EMBBEDDING(Enum):
    MODEL = "Model"
    FOLDER = "Folder"
    
class DETECTION(Enum):
    MODEL = "Model"