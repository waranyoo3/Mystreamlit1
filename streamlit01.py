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

dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])
if st.button("showchart"):
    dt.boxplot()

else:
    st.button("Don't show chart")