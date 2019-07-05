#!/usr/bin/env python
# -*- coding: utf-8 -*-

#PAQUETES

import pandas as pd
import numpy as np 
import glob 
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import datetime as dt
import datetime
import os

import multiprocessing
from multiprocessing import Pool
import time

import funciones_sora as fs

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
    logging.basicConfig(filename = 'plot_extrapol.log',level=logging.INFO)
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
def plot_extrapol(idlcolors=True):
    # inputs acumula radar

    Dt=300.
    nc_basin= '/media/nicolas/maso/Mario/basins/260.nc'
    codigos = [260]
    accum=False;path_tif=None;meanrain_ALL=True;save_bin=False;path_res=None,
    umbral=0.005;rutaNC='/media/nicolas/Home/nicolas/101_RadarClass/'
    path_figs= '/media/nicolas/Home/Jupyter/Soraya/Op_Alarmas/Result_to_web/operacional/acum_radar/'
    
    starts = [fs.round_time(dt.datetime.now()) - pd.Timedelta('30m'),fs.round_time(dt.datetime.now()) ]
    ends = [fs.round_time(dt.datetime.now()), fs.round_time(dt.datetime.now()) + pd.Timedelta('30m')]
    figsnames = ['30minbefore_allradarextent','30minahead_allradarextent']
    
    for start,end,figname in zip(starts,ends,figsnames):
        # Acumula radar.
        print start,end
        print start+ pd.Timedelta('5h'),end+ pd.Timedelta('5h')
        dflol,radmatrix = fs.get_radar_rain(start,end,Dt,nc_basin,codigos,all_radextent=True)
        # inputs fig
        path_figure =  path_figs+figname+'.png'
        rad2plot = radmatrix.T
        window_t='30m'
        #fig
        fs.plot_allradarextent(rad2plot,window_t,idlcolors=idlcolors,path_figure=path_figure,extrapol_axislims=True)
    
#ejecucion
plot_extrapol()
# if __name__ == '__main__':
#     p = multiprocessing.Process(target=plot_extrapol, name="")
#     p.start()
#     time.sleep(60*3) # in seg (seg*min)
#     p.terminate()
#     p.join()
#     print ('plot_extrapol executed')
    