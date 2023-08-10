import streamlit as st
import pickle as p
import numpy as np

# importing the model
pipe= p.load(open('pipee.pkl','rb'))
df= p.load(open('df.pkl','rb'))
# setting the title
st.title('laptop price predictor')

# brand
company=st.selectbox('Brand',df['Company'].unique())

# type of laptop
type=st.selectbox('type',df['TypeName'].unique())
# Ram
ram =  st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])
# Weight
weight= st.number_input('insert the weight')
# touchscreen
touchscreen= st.selectbox('Touchscreen',['No','Yes'])
# IPS
ips = st.selectbox('IPS',['No','Yes'])
# screen size
screen_size = st.number_input('Screen size')
#resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
#cpu
cpu = st.selectbox('Cpu',df['Cpu_brand'].unique())
hdd = st.selectbox('hdd(in gb)',[0,128,256,512,1024,2048])
ssd = st.selectbox('ssd(in GB)',[0,8,128,256,516,1024])
gpu = st.selectbox('GPU',df['Gpu_brand'].unique())
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips=='Yes':
        ips=1
    else:
        ips=0

    x_res=int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2)+(y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    st.title('The predicted price of this configuration is ' + str(int(np.exp(pipe.predict(query)[0]))))


