import glob
import numpy as np
import re
import pyart
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.simplefilter("ignore")
def find_nearest(array,value):
    array=np.asarray(array)
    idx=(np.abs(array-value)).argmin()
    return array[idx]

def find_nearest_idx(array,value):
    array=np.asarray(array)
    idx=(np.abs(array-value)).argmin()
    return idx


aklFiles=glob.glob('AKL/*/*/*/AKL*.RAWSURV*')
aklList=[]
for file in aklFiles:
    m=re.search('(\\d+).RAWSURV',file)
    aklList.append(int(m.group(1)))

aklList.sort()

bopFiles=glob.glob('BOP/*/*/*/BOP*.RAWSURV*')
bopList=[]
for file in bopFiles:
    m=re.search('(\\d+).RAWSURV',file)
    bopList.append(int(m.group(1)))

bopList.sort()
reflectance=[]



# In[ ]:





# In[4]:
import bz2

myMa=[]
for fileNo in range(0,len(aklList)):
        akldt=aklList[fileNo]
        print(akldt)
        if Path(f'radar_{akldt}.png').exists()==False:
            aklDay=str(akldt)[4:6]
            aklMonth=str(akldt)[2:4]
            aklYear=str(2000+int(str(akldt)[0:2]))
            bopdt=str(find_nearest(bopList,akldt))
            bopMonth=str(bopdt)[2:4]
            bopYear=str(2000+int(str(bopdt)[0:2]))
            bopDay=str(bopdt)[4:6]
            file1=f'AKL/{aklYear}/{aklMonth}/{aklDay}/AKL'+str(akldt)+'.RAWSURV.bz2'
            file2=f'BOP/{bopYear}/{bopMonth}/{bopDay}/BOP'+str(bopdt)+'.RAWSURV.bz2'
            fp=open('AKL.RAWSURV','wb') 
            fpb=bz2.open(file1,'rb')
            fp.write(fpb.read())
            fp.close()
            fp=open('BOP.RAWSURV','wb') 
            fpb=bz2.open(file2,'rb')
            fp.write(fpb.read())
            fp.close()
            radar_files=['AKL.RAWSURV', 'BOP.RAWSURV']
            radars = [pyart.io.read(file) for file in radar_files]
            filters = []
            for radar in radars:
                gatefilter = pyart.correct.despeckle.despeckle_field(radar, 'reflectivity',
                                                         threshold=-100, size=20)
                gatefilter.exclude_transition()
                gatefilter.exclude_above('reflectivity', 80)
                filters.append(gatefilter)           
            grid = pyart.map.grid_from_radars(radars, gatefilters=filters,
                                  grid_shape= (5, 250, 350),
                                  grid_limits= ([0.0, 10000.0], [-25000.0, 25000.0], [-35000.0, 35000.0]),
                                  fields=['reflectivity'],
                                  max_refl= 80.,
                                  copy_field_data= True,
                                  grid_origin= (-37.0428, 175.6819),
                                  roi_func= 'dist_beam',
                                  min_radius= 1.0,
                                  h_factor= 1.0,
                                  nb= 1.0,
                                  bsp= 1.0
                                 )                               

            plt.imsave(f'radar_{akldt}.png', grid.fields['reflectivity']['data'][0,:,:], cmap='ocean')


