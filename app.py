import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os

from preprocess import preprocess_image


# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="Hindi OCR",
    page_icon="📝",
    layout="centered"
)

st.title("Hindi Handwritten Character Recognition")


# --------------------------
# LABEL DICTIONARY
# --------------------------

label_map = {

"character_01_ka":"क",
"character_02_kha":"ख",
"character_03_ga":"ग",
"character_04_gha":"घ",
"character_05_kna":"ङ",

"character_06_cha":"च",
"character_07_chha":"छ",
"character_08_ja":"ज",
"character_09_jha":"झ",
"character_10_yna":"ञ",

"character_11_taamatar":"ट",
"character_12_thaa":"ठ",
"character_13_daa":"ड",
"character_14_dhaa":"ढ",
"character_15_adna":"ण",

"character_16_tabala":"त",
"character_17_tha":"थ",
"character_18_da":"द",
"character_19_dha":"ध",
"character_20_na":"न",

"character_21_pa":"प",
"character_22_pha":"फ",
"character_23_ba":"ब",
"character_24_bha":"भ",
"character_25_ma":"म",

"character_26_yaw":"य",
"character_27_ra":"र",
"character_28_la":"ल",
"character_29_waw":"व",

"character_30_motosaw":"श",
"character_31_petchiryakha":"ष",
"character_32_patalosaw":"स",
"character_33_ha":"ह",

"character_34_chhya":"क्ष",
"character_35_tra":"त्र",
"character_36_gya":"ज्ञ",

"digit_0":"०",
"digit_1":"१",
"digit_2":"२",
"digit_3":"३",
"digit_4":"४",
"digit_5":"५",
"digit_6":"६",
"digit_7":"७",
"digit_8":"८",
"digit_9":"९"
}


# --------------------------
# READ CLASS NAMES
# --------------------------

with open("class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]



# --------------------------
# LOAD MODEL
# --------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        46
    )

    model.load_state_dict(
        torch.load(
            "devanagari_resnet18.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# --------------------------
# TRANSFORM
# --------------------------

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((64,64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])


# --------------------------
# FILE UPLOADER
# --------------------------

uploaded = st.file_uploader(
    "Upload Hindi Character Image",
    type=["png","jpg","jpeg"]
)


if uploaded:

    img = Image.open(uploaded)

    st.image(
        img,
        caption="Original Image",
        width=220
    )


    # PREPROCESS

    processed = preprocess_image(img)

    st.image(
        processed,
        caption="Preprocessed Image",
        width=150
    )


    # TRANSFORM

    x = transform(processed)

    x = x.unsqueeze(0)


    # PREDICTION

    with torch.no_grad():

        output = model(x)

        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, dim=1)

        idx = pred.item()


    predicted_class = class_names[idx]

    hindi_character = label_map.get(
        predicted_class,
        predicted_class
    )


    # DISPLAY RESULT

    st.success(
        f"Prediction : {hindi_character}"
    )

    st.write(
        f"Confidence : {confidence.item()*100:.2f}%"
    )