import os
from deepface import DeepFace
import pandas as pd

def recognize_face(target_img_path, datab_path):
    try:
        df_list = DeepFace.find(img_path=target_img_path, db_path=datab_path, model_name="ArcFace", enforce_detection=False)
        if df_list and isinstance(df_list[0], pd.DataFrame) and not df_list[0].empty:
            first_result = df_list[0]
            distance = first_result['distance'].iloc[0]
            recognized_full_path = first_result['identity'].iloc[0]
            similarity_percentage = (1 - distance) * 100
            name = os.path.basename(os.path.dirname(recognized_full_path))
            return name, similarity_percentage
        return None 
    except ValueError as e:
        print(e)