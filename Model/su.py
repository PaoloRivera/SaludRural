
import pickle
from sklearn.metrics import PredictionErrorDisplay
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd



# loading the saved models

diabetes_model = pickle.load(open('saved models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('saved models/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('saved models/parkinsons_model.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('SaludRural AI',
                          
                          ['Sintomas Generales','Predicción de Diabetes',
                           'Predicción de Enfermedad del Corazón',
                           'Predicción de Parkinson'
                           ],
                          icons=['person','activity','heart','cloud'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Predicción de Diabetes'):
    
    # page title
    st.title('Predicción de Diabetes')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Número de Embarazos')
        
    with col2:
        Glucose = st.text_input('Nivel de Glucosa')
    
    with col3:
        BloodPressure = st.text_input('Valor de la Presión Arterial')
    
    with col1:
        SkinThickness = st.text_input('Valor del Grosor de la Piel')
    
    with col2:
        Insulin = st.text_input('Nivel de Insulina')
    
    with col3:
        BMI = st.text_input('Valor del IMC')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Valor de la Función de Pedigrí de Diabetes')
    
    with col2:
        Age = st.text_input('Edad de la Persona')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Resultado de Predicción de Diabetes'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'Es muy posible que tenga diabetes. Se recomienda consultar a un médico para una evaluación más detallada.'
        else:
          diab_diagnosis = 'No cuenta con riesgo de diabetes'

        
    st.success(diab_diagnosis)

if (selected == "Predicción de Parkinson"):
    
    # Título de la página
    st.title("Predicción de la Enfermedad de Parkinson usando ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')

    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Resultado de la predicción de Parkinson"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "Es posible que la persona tenga la enfermedad de Parkinson. Se recomienda consultar a un médico para una evaluación más detallada."
        else:
          parkinsons_diagnosis = "No cuenta con la enfermedad de Parkinson"

        


if (selected == 'Predicción de Enfermedad del Corazón'):
    
    # page title
    st.title('Predicción de Enfermedad del Corazón')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Edad')
        
    with col2:
        sex = st.text_input('Sexo')
        
    with col3:
        cp = st.text_input('Tipos de Dolor de Pecho')
        
    with col1:
        trestbps = st.text_input('Presión Arterial en Reposo')
        
    with col2:
        chol = st.text_input('Colesterol Sérico en mg/dl')
        
    with col3:
        fbs = st.text_input('Azúcar en Sangre en Ayunas > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resultados Electrocardiográficos en Reposo')
        
    with col2:
        thalach = st.text_input('Frecuencia Cardíaca Máxima Alcanzada')
        
    with col3:
        exang = st.text_input('Angina Inducida por Ejercicio')
        
    with col1:
        oldpeak = st.text_input('Depresión del ST Inducida por Ejercicio')
        
    with col2:
        slope = st.text_input('Pendiente del Segmento ST durante el Ejercicio')
        
    with col3:
        ca = st.text_input('Vasos Principales Coloreados por Fluoroscopía')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = defecto fijo; 2 = defecto reversible')

    with col2:
        target = st.text_input('Objetivo')

     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Resultado de la Predicción de Enfermedad del Corazón'):
        # Convert inputs to numeric types before passing to the model
        age = int(age)
        sex = int(sex)
        cp = int(cp)
        trestbps = int(trestbps)
        chol = int(chol)
        fbs = int(fbs)
        restecg = int(restecg)
        thalach = int(thalach)
        exang = int(exang)
        oldpeak = float(oldpeak)  # This is likely a float value
        slope = int(slope)
        ca = int(ca)
        thal = int(thal)
        target = int(target)

        # Pass numeric values to the model for prediction
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])                          
        
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'Es posible que la persona tenga una enfermedad del corazón. Se recomienda consultar a un médico para una evaluación más detallada.'
        else:
            heart_diagnosis = 'No cuenta con riesgo de padecer una enfermedad del corazón.'

    st.success(heart_diagnosis)

# New   
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

l1_display = [symptom.replace('_', ' ') for symptom in l1]
symptom_mapping = {display: internal for display, internal in zip(l1_display, l1)}



disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv(r'dataset/Training.csv')

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)


# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv(r'dataset/Testing.csv')
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)


medicine_dictionary = {
    "Acne": ["Aclear Ointment", "Aclear Capsules"],
    "Allergy": ["Kaeswar Guggula Tablet", "Khadira Rishtha"],
    "Arthritis": ["Jogaraja Guggula", "Rasana Pachana", "Kineaz Capsules"],
    "Bronchial Asthma": ["Talisadi Churna", "Spazex Capsules"],
    "Common Cold": ["Naradiya Laxmivilasa", "Drakshya Rishtha"],
    "Chronic cholestasis": ["Sunthhi Khanda Modaka", "Kutaja Ghanabati"],
    "Covid": ["Ayush 64 Tablet"],
    "Dengue": ["Plate Plus Capsule"],
    "Diabetes": ["Madhu Mehari Bati", "Glucostat Capsule"],
    "Fungal Infection": ["Kaeswar Guggula"],
    "Gastro": ["Gasex Tablet"],
    "Hypertension": ["Ashwagandha"],
    "Hypothyroidism": ["It's better to consult with Doctor"],
    "Jaundice": ["Live 52 DS"],
    "Hypoglycemia": ["It's better to consult with Doctor"],
    "Malaria": ["Ayush 64 Tablet"],
    "Oestroarthritis": ["Ostygen Capsule"],
    "Paralysis": ["Rumartho Gold", "Mahamasa Taila"],
    "Peptic Ulcers": ["Sutin Tablet"],
    "Pneumonia": ["Naradiya Lakhmi Vilas"],
    "Psoriasis": ["It's better to consult with Doctor"],
    "Spondylitis": ["Jogaraja Gugula", "Mahamasa Taila"],
    "Tuberculosis": ["It's better to consult with Doctor"],
    "Typhoid": ["It's better to consult with Doctor"],
    "Urinary Tract": ["Anyolith", "Asocaristha"],
    "Varicose Veins": ["It's better to consult with Doctor"],
    "Alcoholic Hepatitis": ["It's better to consult with Doctor"],
    "Cervical Spondylosis": ["Kineaz Tablets", "Rumartho Gold"],
    "Chicken Pox": ["It's better to consult with Doctor"],
    "Chronic cholestasis": ["Sunthi Khanda Modaka", "Live 52 DS"],
    "Piles": ["Asrsakuthara Rasa", "Pilex"],
    "Drug Reaction": ["It's better to consult with Doctor"],
    "GERD": ["Acilans Capsule"],
    "Gastroenteritis": ["Gasex Tablet", "Sunthi Khanda Modaka"],
    "Heart Attack": ["It's better to consult with Doctor"],
    "Hepatitis A, B, C, D, E": ["It's better to consult with Doctor"],
    "Impetigo": ["It's better to consult with Doctor"],
    "Migraine": ["It's better to consult with Doctor"],
    "Paroxysmal Positional Vertigo": ["Ashwagandha"],
}

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

if (selected == 'Sintomas Generales'):
    
    # Título de la página
    st.title('Predicción de Enfermedades a partir de Síntomas')
    
    # Obtener los síntomas seleccionados en español sin guiones bajos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Symptom1_display = st.selectbox('Síntoma 1', l1_display)
        
    with col2:
        Symptom2_display = st.selectbox('Síntoma 2', l1_display)
    
    with col3:
        Symptom3_display = st.selectbox('Síntoma 3', l1_display)
    
    with col1:
        Symptom4_display = st.selectbox('Síntoma 4', l1_display)
    
    with col2:
        Symptom5_display = st.selectbox('Síntoma 5', l1_display)

    # Mapeo de nombres mostrados a nombres internos con guiones bajos
    Symptom1 = symptom_mapping.get(Symptom1_display, "")
    Symptom2 = symptom_mapping.get(Symptom2_display, "")
    Symptom3 = symptom_mapping.get(Symptom3_display, "")
    Symptom4 = symptom_mapping.get(Symptom4_display, "")
    Symptom5 = symptom_mapping.get(Symptom5_display, "")
    
    # Inicializar l2 a una lista de ceros
    l2 = [0] * len(l1)
    
    # Lista de síntomas seleccionados
    psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]
    
    # Actualizar l2 basado en los síntomas seleccionados
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1
        else:
            l2[k] = 0  # Asegurar que se reinicie a 0 si no está presente
    
    # Entrenar el modelo (considera entrenar y guardar el modelo fuera de la aplicación)
    clf3 = DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)
    
    # Calcular precisión (opcional, puedes eliminar si no es necesario)
    from sklearn.metrics import accuracy_score
    y_pred = clf3.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy*100:.2f}%")
    print(f"Número de correctos: {accuracy_score(y_test, y_pred, normalize=False)}")
    
    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]
    
    # Obtener las medicinas recomendadas
    medicines = medicine_dictionary.get(disease[predicted], ["No hay medicinas recomendadas para esta enfermedad. Se recomienda consultar a un médico."])
    
    if st.button('Resultado de Predicción'):
        st.success(f"El resultado de la predicción es: **{disease[predicted]}**")
        st.info("Las medicinas recomendadas son:")
        for medicine in medicines:
            st.success(f"- {medicine}")

st.title('SaludRural AI')