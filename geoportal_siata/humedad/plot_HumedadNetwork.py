#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cprv1.cprv1 as cprv1

import datetime
import pandas as pd
import numpy as np
import multiprocessing
import time
#paquetes sora
import alarmas as al
import glob
import json
import datetime as dt
import cprsora.humedad as hm
from cpr import cpr as cpr_1

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
    logging.basicConfig(filename = 'plot_HumedadNetwork.log',level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        start = time.time()
        f = orig_func(*args,**kwargs)
        date = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        took = time.time()-start
        log = '%s:%s:%.1f sec'%(date,orig_func.__name__,took)
        print log
        logging.info(log)
        return f
    return wrapper

@logger
def plot_HNetwork(selfH,codes2fix,depths2drop):
    queryH=cprv1.SqlDb(**selfH.local_server).read_sql('select codigo from estaciones_estaciones where clase="H" and estado in ("A","P")')
    codes=np.array(queryH['codigo'])
    ruta_figs='/media/nicolas/Home/Jupyter/Soraya/Op_Alarmas/Result_to_web/figs_operacionales/'
    
    #drop depth list
    drop_depths_list = [[None] for code in range(codes.size)]
    for code2fix,depth in zip(codes2fix,depths2drop):
        ind = np.where(codes==code2fix)[0][0]
        drop_depths_list[ind] = depth
    
    for code,drop_depth in zip(codes,drop_depths_list):
        self= hm.Humedad(codigo=code)
        end = dt.datetime.now()
        starts  = [(end - dt.timedelta(hours=3)), (end - dt.timedelta(hours=24)),
                   (end - dt.timedelta(hours=72)),(end - dt.timedelta(days=30)) ]
        for start in starts:
            #consulta pluvio
            P = cprv1.SqlDb(**self.remote_server1).read_sql("select fecha,hora,p1/1000,p2/1000 from datos where cliente='%s' and (((fecha>'%s') or (fecha='%s' and hora>='%s')) and ((fecha<'%s') or (fecha='%s' and hora<='%s')))"%(int(self.info.get('pluvios')),start.strftime('%Y-%m-%d'),start.strftime('%Y-%m-%d'),start.strftime('%H:%M:%S'),end.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),end.strftime('%H:%M:%S')))
            P.columns=['fecha','hora','p1','p2']
            nulls = np.where(P[['fecha']]['fecha'].isnull() == True)[0]
            P= P.drop(nulls)
            dates = [P['fecha'][i].strftime('%Y-%m-%d') +' '+str(P['hora'][i]).split(' ')[-1][:-3] for i in P.index]
            P.index = pd.to_datetime(dates)
            P['fecha_hora']= dates
            P = P.drop_duplicates(['fecha_hora'],keep='first').asfreq('1T')
            P = P.drop(['fecha','hora','fecha_hora'],axis=1)

            self.plot_Humedad2Webpage(start,end,P,ruta_figs=ruta_figs,drop_depth=drop_depth)
    print 'Se ejecutan las graficas operacionales de la red de humedad'

#EJECUCION

# self = cprv1.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
selfH = hm.Humedad(codigo=235)
codes2fix = [235]
depths2drop = [['3']]

# plot_HNetwork(codes2fix,depths2drop)

# Plots de Red de Humedad,ejecuta plots en paralelo   
if __name__ == '__main__':
    p = multiprocessing.Process(target=plot_HNetwork,args=(selfH,codes2fix,depths2drop,), name="r")
    p.start()
    time.sleep(250) # wait near 5 minutes to kill process
    p.terminate()
    p.join()
    print 'plot_HNetwork executed'
