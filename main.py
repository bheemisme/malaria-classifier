import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from model import model

def uploader():
    uploaded_file = st.file_uploader(
                    label='Upload the image', 
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=False,
                    key='image-uploader'
                )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        classname, prob =model.predict(image)
        st.markdown(f'#### Class: {classname}')
        st.markdown(f'#### Probability: {prob}')

def app():
    st.title('Malaria Classifier')
    df = pd.read_csv('./classification_report.csv')
    fig = plt.figure()
    sns.barplot(data=df, x='metrics', y='values')

    col1, col2 = st.columns(2)
    with col1:
        st.image('./images/results.png', use_column_width=True)
    
    with col2:
        st.pyplot(fig)

    uploader()


if __name__ == '__main__':
    app()
