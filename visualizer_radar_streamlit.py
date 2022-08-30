import streamlit as st
import pandas as pd
import numpy as np
import glob
import numpy as np
import re
import pyart
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import st_folium
import folium
import datetime

endd=datetime.datetime.now().replace(second=0,microsecond=0)
discard = datetime.timedelta(minutes=endd.minute % 5)
endd -= discard
aklFiles=glob.glob('radar_*.png')
aklList=[]
for file in aklFiles:
    m=re.search('radar_(\\d+).png',file)
    aklList.append(int(m.group(1)))

 
st.title('Real-time Radar Visualizer with streamlit')
startd=datetime.datetime(2022, 8, 1, hour=0, minute=0, second=0)
querydt=st.slider('Date/time',startd,endd,format='D MMM YYYY, H:mm:ss',step=datetime.timedelta(minutes=7.5))

def find_nearest(array,value):
    array=np.asarray(array)
    idx=(np.abs(array-value)).argmin()
    return array[idx]

def find_nearest_idx(array,value):
    array=np.asarray(array)
    idx=(np.abs(array-value)).argmin()
    return idx

querystr=querydt.strftime("%y%m%d%H%M%S")
st.write(querystr)
enddt=int(querystr)
endIdx=find_nearest_idx(aklList,enddt)
akldt=aklList[endIdx]
tmap = folium.Map(location=[-37.0428, 175.6819], zoom_start=11)
img = folium.raster_layers.ImageOverlay(
        name="Rain Radar Reflectance",
        image=f'radar_{akldt}.png',
        bounds=[[-36.8428, 175.30819],[-37.2428, 175.9819]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
    )
img.add_to(tmap)
folium.LayerControl().add_to(tmap)
# call to render Folium map in Streamlit
st_data = st_folium(tmap, width = 725)

