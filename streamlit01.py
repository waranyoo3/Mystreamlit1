import streamlit as st
import pandas as pd


st.title('waranyoo')
col1,col2 = st.columns(2)
with col1: 
    st.image("./pic/DSCF0584.jpg")
with col2:
    st.image("./pic/DSCF0584.jpg")
        
st.header("วรัญญู จุ๊กกรู้ 5555")
st.subheader('pee')
st.image ("./pic/css3logo.jpg")

dt=pd.read_csv('data/iris.csv')
st.write(dt.head(10))

st.button("showchart")