import matplotlib 
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager

font_dirs = ['/media/nicolas/Home/Jupyter/Sebastian/AvenirLTStd-Book']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

matplotlib.rcParams['font.family'] = 'Avenir LT Std'
matplotlib.rcParams['font.size']=14


# import cprv1.cprv1 as cprv1
import pandas as pd
import numpy as np
import datetime as dt
import json
import wmf.wmf as wmf
import multiprocessing
from multiprocessing import Pool
import time
# import datetime

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
    logging.basicConfig(filename = 'SHOp_Ejecucion_v2.log',level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        import datetime
        start = time.time()
        f = orig_func(*args,**kwargs)
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        took = time.time()-start
        log = '%s:%s:%.1f sec'%(date,orig_func.__name__,took)
        print log
        logging.info(log)
        return f
    return wrapper

@logger
def model_trigger(configfile,rutafig):
    import os
    import datetime
    #lee la ruta.
    ConfigList= SHop.get_rutesList(configfile)
    path_op= SHop.get_ruta(ConfigList,'ruta_proj_op')

    if any(np.array(os.listdir(path_op)) == 'flag2run') == True:
        datenow = pd.to_datetime(dt.datetime.strftime(SHop.round_time(dt.datetime.now()), '%Y-%m-%d %H:%M'))
        # fecha real para pruebas.
        date2run = datenow - pd.Timedelta('77 days') #####################CHANGE!

        #se evalua la edad del archivo
        file_date = SHop.round_time(dt.datetime.fromtimestamp(os.path.getctime(path_op+'flag2run')))
        file_age = np.abs((datenow - pd.to_datetime(file_date)).total_seconds()/60.0) # mins
        print 'Flag file exist %s mins ago, so run between with date: %s'%(file_age, date2run)
        #running between 12h back till 30m after date2run.
        SHop.run_model_op(ConfigList,file_age,date2run,rutafig=rutafig)
    else:
        print 'Flag file does not exist'
        pass

#Ejecucion
rutafig = '/media/nicolas/Home/Jupyter/Soraya/Op_Alarmas/Result_to_web/SH_Op/HOp_AMVA90m/'
configfile='/media/nicolas/Home/Jupyter/Soraya/Alarmas_last/03_modelo/Op_E260_60m_py3/configfile.md'

model_trigger(configfile,rutafig)