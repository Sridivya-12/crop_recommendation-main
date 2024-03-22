import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.columns([2,2])
    
    with col1: 
        with st.expander(" Expert System: Crop-Recommendation", expanded=True):
            st.image("es.jpg", caption="Expert System", use_column_width=True)
            st.write("""
            Here our "Knowledge Base" is prepared with "Crop_Recommendation.csv" file. This data has been collected from "Farms" and "Human Experts".
            An "Inference Engine" is been prepared using Machine learning and some other technologies. This Interface is developed with "Streamlit".
            Now Any Non-Expert can interact with Expert System and can get the suggestions like getting suggesions from a Human Expert.

            """)
        '''
        ## How does it work ❓ 
        Fill the details shown then Expert System will give you a crop recommendation.  
        '''


    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen ratio to soil", 1,10000)
        P = st.number_input("Phosporus ratio to soil", 1,10000)
        K = st.number_input("Potassium ratio to soil", 1,10000)
        temp = st.number_input("Temperature degree C",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Give me Recommendation'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} will be suitable for your farm.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning("Note: This A.I application is for educational purposes only. Check the source code [here](https://github.com/Jiten15/crop_recommendation)")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()