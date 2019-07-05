# import cprv1.cprv1 as cprv1
import datetime as dt
import pandas as pd
import os
import glob
import numpy as np
import json
import wmf.wmf as wmf
import multiprocessing
from multiprocessing import Pool
import time

import SH_operacional as SHop

def logger(orig_func):
    '''logging decorator, alters function passed as argument and creates
    log file. (contains function time execution)
    Parameters
    ----------
    orig_func : function to pass into decorator
    Returns
    -------
    log file
    '''
    import logging
    from functools import wraps
    import time
    logging.basicConfig(filename = 'assess_RadRisk.log',level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        start = time.time()
        f = orig_func(*args,**kwargs)
        date = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        took = time.time()-start
        log = '%s:%s:%.1f sec'%(date,orig_func.__name__,took)
        print (log)
        logging.info(log)
        return f
    return wrapper

@logger
def assess_flagfile(dfacum_past,dfacum_ahead,df_ref):
    for ID in df_ref.dropna().sort_index().index: # to each station with a threshold assigned         
        #assess if 3hacum is >= threshold and 3hacum will increaseinnext30m. if dfacum_ahead[ID][-1]=nan.. 'or' rather than '&'
        print (ID)
        if ((dfacum_past[ID] >= df_ref['threshold'].loc[ID]) & (dfacum_ahead[ID] > dfacum_past[ID])) == True or ((dfacum_past[ID] >= df_ref['threshold'].loc[ID]) or (dfacum_ahead[ID] > dfacum_past[ID])) == True:
            #look for the flag file
            if 'flag2run' in os.listdir(df_ref['proj_paths'].loc[ID]):
                #si existe no pasa nada
                datenow=pd.to_datetime(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M'))
                file_date= SHop.round_time(dt.datetime.fromtimestamp(os.path.getctime(df_ref['proj_paths'].loc[ID]+'flag2run')))
                file_age = np.abs((datenow - pd.to_datetime(file_date)).total_seconds()/60.0)
                print ('Flag file age: '+str(file_age)+' min.')
            else:
                #si no existe lo crea
                f = open(df_ref['proj_paths'].loc[ID]+'flag2run','w')
                f.close()        
                print (df_ref['proj_paths'].loc[ID]+'flag2run' + ' now exists.')

        #if not, look for the flag file and asses its age in all df_ref['proj_paths'] if they exist.
        else:
            print ('%s basin has not reached its mean rainfall threshold.'%(ID))
            # look for the flag file
            if 'flag2run' in os.listdir(df_ref['proj_paths'].loc[ID]):
                # if it exists, asses its age i n  mins
                datenow=pd.to_datetime(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M'))
                file_date= SHop.round_time(dt.datetime.fromtimestamp(os.path.getctime(df_ref['proj_paths'].loc[ID]+'flag2run')))
                file_age = np.abs((datenow - pd.to_datetime(file_date)).total_seconds()/60.0)
                print ('Flag file age: '+str(file_age)+' min.')
                #if is older than 6h - 360min, remomve it
                if file_age > 360: 
                    os.remove (df_ref['proj_paths'].loc[ID]+'flag2run')
                    print ('Flag file has been removed')
                #if it's younger, nothing happens
                else:
                    pass
                    print ('Flag file stays.')
            #if it doesn't exist, nothing happens
            else:
                print ('There is no flag file.')
                pass

#####EJECUCION

#Radar rainfall.
datenow=pd.to_datetime(dt.datetime.strftime(SHop.round_time(dt.datetime.now()), '%Y-%m-%d %H:%M'))
Dt = 300.
cuenca = '/media/nicolas/maso/Mario/basins/260.nc'
# every basin with mask.
masks_list = glob.glob('/media/nicolas/maso/Mario/mask/mask_*.tif')
codigos = np.sort([int(path.split('_')[-1].split('.')[0]) for path in masks_list])
start = datenow - pd.Timedelta('3h')
posterior = datenow + pd.Timedelta('30m')

dfrad = SHop.get_radar_rain(start,posterior,Dt,cuenca,codigos,verbose=False)

#INPUTS
dfacum_past =  dfrad.loc[start:datenow].sum()
dfacum_ahead =  dfrad.loc[datenow:posterior].sum()
#DF thresholds
path_umbrales= '/media/nicolas/maso/Soraya/SHOp_files/umbRadMean_Basins.csv'
df_ref = pd.read_csv(path_umbrales,index_col=0)

#FUNCION CRONEADA
assess_flagfile(dfacum_past,dfacum_ahead,df_ref)
