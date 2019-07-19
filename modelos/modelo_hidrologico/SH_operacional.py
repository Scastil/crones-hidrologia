#!/usr/bin/env python
# -- coding: utf-8 --

import pandas as pd
from wmf import wmf
import numpy as np 
import glob 
import pylab as pl
import json
import MySQLdb
import csv
import matplotlib
import matplotlib.font_manager
from datetime import timedelta
import datetime as dt
import pickle
import matplotlib.dates as mdates
import netCDF4
import textwrap
# from cpr import cpr
from multiprocessing import Pool

import os
import datetime
from cprv1 import cprv1

#---------------
#Funciones base.
#---------------

def get_rutesList(rutas):
    ''' Abre el archivo de texto en la ruta: rutas, devuelve una lista de las lineas de ese archivo.
        Funcion base.
        #Argumentos
        rutas: string, path indicado.
    '''
    f = open(rutas,'r')
    L = f.readlines()
    f.close()
    return L


def round_time(date = dt.datetime.now(),round_mins=5):
    '''
    Rounds datetime object to nearest 'round_time' minutes.
    If 'dif' is < 'round_time'/2 takes minute behind, else takesminute ahead.
    Parameters
    ----------
    date         : date to round
    round_mins   : round to this nearest minutes interval
    Returns
    ----------
    datetime object rounded, datetime object
    '''    
    dif = date.minute % round_mins

    if dif <= round_mins/2:
        return dt.datetime(date.year, date.month, date.day, date.hour, date.minute - (date.minute % round_mins))
    else:
        return dt.datetime(date.year, date.month, date.day, date.hour, date.minute - (date.minute % round_mins)) + dt.timedelta(minutes=round_mins)
    
def set_modelsettings(ConfigList):
    ruta_modelset = get_ruta(ConfigList,'ruta_modelset')
    # model settings  Json
    with open(ruta_modelset, 'r') as f:
        model_set = json.load(f)
    # Model set
    wmf.models.retorno = model_set['retorno']
    wmf.models.show_storage = model_set['show_storage']
    wmf.models.separate_fluxes = model_set['separate_fluxes']
    wmf.models.dt = model_set['dt']
    
def time_windows(date,warming_window='4h',windows = ['3h','6h','12h']):
    #ventanas de tiempo.
    #date =  pd.to_datetime(dt.datetime.now().strftime(%Y-%m-%d %H:%M))
    date = round_time(pd.to_datetime(date))
    starts = [date - pd.Timedelta(window) for window in windows]
    starts_m = [start- pd.Timedelta(warming_window) for start in starts] # warming window
    end = date + pd.Timedelta('30m') # extrapol
#     end = date + pd.Timedelta('3h')     
    return starts,starts_m,end,windows
    
#-----------------------------------
#-----------------------------------
#Funciones de lectura del configfile
#-----------------------------------
#-----------------------------------

def get_ruta(RutesList, key):
    ''' Busca en una lista 'RutesList' la linea que empieza con el key indicado, entrega rutas.
        Funcion base.
        #Argumentos
        RutesList: Lista que devuelve la funcion en este script get_rutesList()
        key: string, key indicado para buscar que linea en la lista empieza con el.
    '''
    if any(i.startswith('- **'+key+'**') for i in RutesList):
        for i in RutesList:
            if i.startswith('- **'+key+'**'):
                return i.split(' ')[-1][:-1]
    else:
        return 'Aviso: no existe linea con el key especificado'
    
def get_line(RutesList, key):
    ''' Busca en una lista 'RutesList' la linea que empieza con el key indicado, entrega lineas.
        Funcion base.
        #Argumentos
        RutesList: Lista que devuelve la funcion en este script get_rutesList()
        key: string, key indicado para buscar que linea en la lista empieza con el.
    '''
    if any(i.startswith('- **'+key+'**') for i in RutesList):
        for i in RutesList:
            if i.startswith('- **'+key+'**'):
                return i[:-1].split(' ')[2:]
    else:
        return 'Aviso: no existe linea con el key especificado'

def get_modelPlot(RutesList, PlotType = 'Qsim_map'):
    ''' #Devuelve un diccionario con la informacion de la tabla Plot en el configfile.
        #Funcion operacional.
        #Argumentos:
        - RutesList= lista, es el resultado de leer el configfile con al.get_ruteslist.
        - PlotType= boolean, tipo del plot? . Default= 'Qsim_map'.
    '''
    for l in RutesList:
        key = l.split('|')[1].rstrip().lstrip()
        if key[3:] == PlotType:
            EjecsList = [i.rstrip().lstrip() for i in l.split('|')[2].split(',')]
            return EjecsList
    return key

def get_modelPars(RutesList):
    ''' #Devuelve un diccionario con la informacion de la tabla Calib en el configfile.
        #Funcion operacional.
        #Argumentos:
        - RutesList= lista, es el resultado de leer el configfile con al.get_ruteslist.
    '''
    DCalib = {}
    for l in RutesList:
        c = [float(i) for i in l.split('|')[3:-1]]
        name = l.split('|')[2]
        DCalib.update({name.rstrip().lstrip(): c})
    return DCalib

def get_modelPaths(List):
    ''' #Devuelve un diccionario con la informacion de la tabla Calib en el configfile.
        #Funcion operacional.
        #Argumentos:
        - RutesList= lista, es el resultado de leer el configfile con al.get_ruteslist.
    '''
    DCalib = {}
    for l in List:
        c = [i for i in l.split('|')[3:-1]]
        name = l.split('|')[2]
        DCalib.update({name.rstrip().lstrip(): c[0]})
    return DCalib

def get_modelStore(RutesList):
    ''' #Devuelve un diccionario con la informacion de la tabla Store en el configfile.
        #Funcion operacional.
        #Argumentos:
        - RutesList= lista, es el resultado de leer el configfile con al.get_ruteslist.
    '''
    DStore = {}
    for l in RutesList:
        l = l.split('|')
        DStore.update({l[1].rstrip().lstrip():
            {'Nombre': l[2].rstrip().lstrip(),
            'Actualizar': l[3].rstrip().lstrip(),
            'Tiempo': float(l[4].rstrip().lstrip()),
            'Condition': l[5].rstrip().lstrip(),
            'Calib': l[6].rstrip().lstrip(),
            'BackSto': l[7].rstrip().lstrip(),
            'Slides': l[8].rstrip().lstrip()}})
    return DStore

def get_modelStoreLastUpdate(RutesList):
    ''' #Devuelve un diccionario con la informacion de la tabla Update en el configfile.
        #Funcion operacional.
        #Argumentos:
        - RutesList= lista, es el resultado de leer el configfile con al.get_ruteslist.
    '''
    DStoreUpdate = {}
    for l in RutesList:
        l = l.split('|')
        DStoreUpdate.update({l[1].rstrip().lstrip():
            {'Nombre': l[2].rstrip().lstrip(),
            'LastUpdate': l[3].rstrip().lstrip()}})
    return DStoreUpdate

def get_ConfigLines(RutesList, key, keyTable = None, PlotType = None):
    ''' #Devuelve un diccionario con la informacion de las tablas en el configfile: Calib, Store, Update, Plot.
        #Funcion operacional.
        #Argumentos:
        - RutesList= lista, es el resultado de leer el configfile con al.get_ruteslist.
        - key= string, palabra clave de la tabla que se quiere leer. Puede ser: -s,-t.
        - Calib_Storage= string, palabra clave de la tabla que se quiere leer. Puede ser: Calib, Store, Update, Plot.
        - PlotType= boolean, tipo del plot? . Default= None.
    '''
    List = []
    for i in RutesList:
        if i.startswith('|'+key) or i.startswith('| '+key):
            List.append(i)
    if len(List)>0:
        if keyTable == 'Pars':
            return get_modelPars(List)
        if keyTable == 'Paths':
            return get_modelPaths(List)
        if keyTable == 'Store':
            return get_modelStore(List)
        if keyTable == 'Update':
            return get_modelStoreLastUpdate(List)
        if keyTable == 'Plot':
            return get_modelPlot(List, PlotType=PlotType)
        return List
    else:
        return 'Aviso: no se encuentran lineas con el key de inicio especificado.'

#-------------------------------------
#-------------------------------------
#Funciones intermedias para el montaje
#-------------------------------------
#-------------------------------------
    
# nc_basin = '/media/nicolas/maso/Soraya/nc_simul/AMVA_PteGab_90m_v221_py2.nc'
# cu = wmf.SimuBasin(rute=nc_basin)
# ests =  np.array([169,106,179,94 ,93 ,99,359,346,140,342,260])#,236,182,238,128,239]
# selfn = cprv1.Nivel(user='sora',passwd='12345',codigo=260)
# coords = selfn.infost.loc[ests][['longitud','latitud']]
# tramos =  get_reachID_from_coord(cu,coords)
# #se corrige est 94 que quedó sobre un afluente y no sobre el tramo del rio
# tramos.loc[94] = 420
# tramos.to_csv('lol.csv')
    
#-----------------------------------
#-----------------------------------
#Funciones generacion de radar
#-----------------------------------
#-----------------------------------

def file_format(start,end):
    '''
    Returns the file format customized for siata for elements containing
    starting and ending point
    Parameters
    ----------
    start        : initial date
    end          : final date
    Returns
    ----------
    file format with datetimes like %Y%m%d%H%M
    Example
    ----------
    '''
    start,end = pd.to_datetime(start),pd.to_datetime(end)
    format = '%Y%m%d%H%M'
    return '%s-%s'%(start.strftime(format),end.strftime(format))

def hdr_to_series(path):
    '''
    Reads hdr rain files and converts it into pandas Series
    Parameters
    ----------
    path         : path to .hdr file
    Returns
    ----------
    pandas time Series with mean radar rain
    '''
    s =  pd.read_csv(path,skiprows=5,usecols=[2,3]).set_index(' Fecha ')[' Lluvia']
    s.index = pd.to_datetime(list(map(lambda x:x.strip()[:10]+' '+x.strip()[11:],s.index)))
    return s

def hdr_to_df(path):
    '''
    Reads hdr rain files and converts it into pandas DataFrame
    Parameters
    ----------
    path         : path to .hdr file
    Returns
    ----------
    pandas DataFrame with mean radar rain
    '''
    if path.endswith('.hdr') != True:
        path = path+'.hdr'
    df = pd.read_csv(path,skiprows=5).set_index(' Fecha ')
    df.index = pd.to_datetime(list(map(lambda x:x.strip()[:10]+' '+x.strip()[11:],df.index)))
    df = df.drop('IDfecha',axis=1)
    df.columns = ['record','mean_rain']
    return df

def bin_to_df(path,ncells,start=None,end=None,**kwargs):
    '''
    Reads rain fields (.bin) and converts it into pandas DataFrame
    Parameters
    ----------
    path         : path to .hdr and .bin file
    start        : initial date
    end          : final date
    Returns
    ----------
    pandas DataFrame with mean radar rain
    Note
    ----------
    path without extension, ejm folder_path/file not folder_path/file.bin,
    if start and end is None, the program process all the data
    '''
    start,end = pd.to_datetime(start),pd.to_datetime(end)
    records = df['record'].values
    rain_field = []
    for count,record in enumerate(records):
        if record != 1:
            rain_field.append(wmf.models.read_int_basin('%s.bin'%path,record,ncells)[0]/1000.0)
            count = count+1
#             format = (count*100.0/len(records),count,len(records))
        else:
            rain_field.append(np.zeros(ncells))
    return pd.DataFrame(np.matrix(rain_field),index=df.index)

def file_format_date_to_datetime(string):
    '''
    Transforms string in file_format like string to datetime object
    Parameters
    ----------
    string         : string object in file_format like time object
    Returns
    ----------
    datetime object
    Example
    ----------
    In : file_format_date_to_datetime('201707141212')
    Out: Timestamp('2017-07-14 12:12:00')
    '''
    format = (string[:4],string[4:6],string[6:8],string[8:10],string[10:12])
    return pd.to_datetime("%s-%s-%s %s:%s"%format)

def file_format_to_variables(string):
    '''
    Splits file name string in user and datetime objects
    Parameters
    ----------
    string         : file name
    Returns
    ----------
    (user,start,end) - (string,datetime object,datetime object)
    '''
    string = string[:string.find('.')]
    start,end,codigo,user = list(x.strip() for x in string.split('-'))
    start,end = file_format_date_to_datetime(start),file_format_date_to_datetime(end)
    return start,end,codigo,user

def check_rain_files(start,end,code,rain_path):#code, rain_path +
    '''
    Finds out if rain data has already been processed
    start        : initial date
    end          : final date
    Returns
    ----------
    file path or None for no coincidences
    '''
    def todate(date):
        return pd.to_datetime(pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')) 
    start,end = todate(start),todate(end)
    files = os.listdir(rain_path) 
    if files:
        for file in files:
            comienza,finaliza,codigo,usuario = file_format_to_variables(file)
            if (comienza==start) and (finaliza==end) and (codigo==code): #codigo - #code+
                file =  file[:file.find('.')]
                print(file)
                break
            else:
                file = None
    else:
        file = None
    return file

def get_radar_rain(start,end,Dt,cuenca,codigos,accum=False,path_tif=None,all_radextent=False,meanrain_ALL=True,complete_naninaccum=False,save_bin=False,
               save_class = False,path_res=None,umbral=0.005,rutaNC='/media/nicolas/Home/nicolas/101_RadarClass/',verbose=True):

    '''
    Read .nc's file forn rutaNC:101Radar_Class within assigned period and frequency.

    0. It divides by 1000.0 and converts from mm/5min to mm/h.
    1. Get mean radar rainfall in basins assigned in 'codigos' for finding masks, if the mask exist.
    2. Write binary files if is setted.
    - Cannot do both 1 and 2.
    - To saving binary files (2) set: meanrain_ALL=False, save_bin=True, path_res= path where to write results, 
      len('codigos')=1, nc_path aims to the one with dxp and simubasin props setted.

    Parameters
    ----------
    start:        string, date&time format %Y-%m%-d %H:%M, local time.
    end:          string, date&time format %Y-%m%-d %H:%M, local time.
    Dt:           float, timedelta in seconds. For this function it should be lower than 3600s (1h).
    cuenca:       string, simubasin .nc path with dxp and format from WMF. It should be 260 path if whole catchment analysis is needed, or any other .nc path for saving the binary file.
    codigos:       list, with codes of stage stations. Needed for finding the mask associated to a basin.
    rutaNC:       string, path with .nc files from radar meteorology group. Default in amazonas: 101Radar_Class

    Optional Parameters
    ----------
    accum:        boolean, default False. True for getting the accumulated matrix between start and end.
                  Change returns: df,rvec (accumulated)
    path_tif:     string, path of tif to write accumlated basin map. Default None.
    all_radextent:boolean, default False. True for getting the accumulated matrix between start and end in the
                  whole radar extent. Change returns: df,radmatrix.
    meanrain_ALL: boolean, defaul True. True for getting the mean radar rainfall within several basins which mask are defined in 'codigos'.
    save_bin:     boolean, default False. True for saving .bin and .hdr files with rainfall and if len('codigos')=1.
    save_class:  boolean,default False. True for saving .bin and .hdr for convective and stratiform classification. Applies if len('codigos')=1 and save_bin = True.
    path_res:     string with path where to write results if save_bin=True, default None.
    umbral:       float. Minimum umbral for writing rainfall, default = 0.005.

    Returns
    ----------
    - df whith meanrainfall of assiged codes in 'codigos'.
    - df,rvec if accum = True.
    - df,radmatrix if all_radextent = True.
    - save .bin and .hdr if save_bin = True, len('codigos')=1 and path_res=path.

    '''

    start,end = pd.to_datetime(start),pd.to_datetime(end)
    #hora UTC
    startUTC,endUTC = start + pd.Timedelta('5 hours'), end + pd.Timedelta('5 hours')
    fechaI,fechaF,hora_1,hora_2 = startUTC.strftime('%Y-%m-%d'), endUTC.strftime('%Y-%m-%d'),startUTC.strftime('%H:%M'),endUTC.strftime('%H:%M')

    #Obtiene las fechas por dias para listar archivos por dia
    datesDias = pd.date_range(fechaI, fechaF,freq='D')

    a = pd.Series(np.zeros(len(datesDias)),index=datesDias)
    a = a.resample('A').sum()
    Anos = [i.strftime('%Y') for i in a.index.to_pydatetime()]

    datesDias = [d.strftime('%Y%m%d') for d in datesDias.to_pydatetime()]

    #lista los .nc existentes de ese dia: rutas y fechas del nombre del archivo
    ListDatesinNC = []
    ListRutas = []
    for d in datesDias:
        try:
            L = glob.glob(rutaNC + d + '*.nc')
            ListRutas.extend(L)
            for i in L: # incluye fechas de extrapol en caso de que fechaF inluya fechas del futuro.
                if i[-11:].endswith('extrapol.nc'):
                    ListDatesinNC.append(i.split('/')[-1].split('_')[0])
                else:
                    ListDatesinNC.append(i.split('/')[-1].split('_')[0])
        except:
            print ('mierda')

    # Organiza las listas de dias y de rutas
    ListDatesinNC.sort()
    ListRutas.sort()
    #index con las fechas especificas de los .nc existentes de radar
    datesinNC = [dt.datetime.strptime(d,'%Y%m%d%H%M') for d in ListDatesinNC]
    datesinNC = pd.to_datetime(datesinNC)


    #Obtiene el index con la resolucion deseada, en que se quiere buscar datos existentes de radar, 
    textdt = '%d' % Dt
    #Agrega hora a la fecha inicial
    if hora_1 != None:
            inicio = fechaI+' '+hora_1
    else:
            inicio = fechaI
    #agrega hora a la fecha final
    if hora_2 != None:
            final = fechaF+' '+hora_2
    else:
            final = fechaF
    datesDt = pd.date_range(inicio,final,freq = textdt+'s')

    #Obtiene las posiciones de acuerdo al dt para cada fecha, si no hay barrido en ese paso de tiempo se acumula 
    #elbarrido inmediatamente anterior.
    PosDates = []
    pos1 = [0]
    for d1,d2 in zip(datesDt[:-1],datesDt[1:]):
            pos2 = np.where((datesinNC<d2) & (datesinNC>=d1))[0].tolist()
            if len(pos2) == 0 and complete_naninaccum == True:
                    pos2 = pos1
            elif complete_naninaccum == True:
                    pos1 = pos2
            elif len(pos2) == 0:
                    pos2=[]
            PosDates.append(pos2)

    # acumular dentro de la cuenca.
    cu = wmf.SimuBasin(rute= cuenca)
    if save_class:
        cuConv = wmf.SimuBasin(rute= cuenca)
        cuStra = wmf.SimuBasin(rute= cuenca)
    # paso a hora local
    datesDt = datesDt - dt.timedelta(hours=5)
    datesDt = datesDt.to_pydatetime()
    #Index de salida en hora local
    rng= pd.date_range(start.strftime('%Y-%m-%d %H:%M'),end.strftime('%Y-%m-%d %H:%M'), freq=  textdt+'s')
    df = pd.DataFrame(index = rng,columns=codigos)

    #accumulated in basin
    if accum:
        rvec_accum = np.zeros(cu.ncells)
        rvec = np.zeros(cu.ncells)
        dfaccum = pd.DataFrame(np.zeros((cu.ncells,rng.size)).T,index = rng)
    else:
        pass

    #all extent
    if all_radextent:
        radmatrix = np.zeros((1728, 1728))

    # print ListRutas
    for dates,pos in zip(datesDt[1:],PosDates):
            rvec = np.zeros(cu.ncells)   
            if save_class:
                rConv = np.zeros(cu.ncells, dtype = int)   
                rStra = np.zeros(cu.ncells, dtype = int)   
            try:
                    #se lee y agrega lluvia de los nc en el intervalo.
                    for c,p in enumerate(pos):
                            #lista archivo leido
                            if verbose:
                                print ListRutas[p]
                            #Lee la imagen de radar para esa fecha
                            g = netCDF4.Dataset(ListRutas[p])
                            #if all extent
                            if all_radextent:
                                radmatrix += g.variables['Rain'][:].T/(((len(pos)*3600)/Dt)*1000.0) 
                            #on basins --> wmf.
                            RadProp = [g.ncols, g.nrows, g.xll, g.yll, g.dx, g.dx]
                            #Agrega la lluvia en el intervalo 
                            rvec += cu.Transform_Map2Basin(g.variables['Rain'][:].T/(((len(pos)*3600)/Dt)*1000.0),RadProp)
                            if save_class:
                                ConvStra = cu.Transform_Map2Basin(g.variables['Conv_Strat'][:].T, RadProp)
                                # 1-stra, 2-conv
                                rConv = np.copy(ConvStra) 
                                rConv[rConv == 1] = 0; rConv[rConv == 2] = 1
                                rStra = np.copy(ConvStra)
                                rStra[rStra == 2] = 0 
                                rvec[(rConv == 0) & (rStra == 0)] = 0
                                Conv[rvec == 0] = 0
                                Stra[rvec == 0] = 0
                            #Cierra el netCDF
                            g.close()
            except:
                    print ('error - zero field ')
                    if accum:
                        rvec_accum += np.zeros(cu.ncells)
                        rvec = np.zeros(cu.ncells)
                    else:
                        rvec = np.zeros(cu.ncells) 
                        if save_class:
                            rConv = np.zeros(cu.ncells)
                            rStra = np.zeros(cu.ncells)
                    if all_radextent:
                        radmatrix += np.zeros((1728, 1728))
            #acumula dentro del for que recorre las fechas
            if accum:
                rvec_accum += rvec
                dfaccum.loc[dates.strftime('%Y-%m-%d %H:%M:%S')]= rvec
            else:
                pass
            # si se quiere sacar promedios de lluvia de radar en varias cuencas definidas en 'codigos'
            if meanrain_ALL:
                mean = []
                #para todas
                for codigo in codigos:
                    if 'mask_%s.tif'%(codigo) in os.listdir('/media/nicolas/maso/Mario/mask/'):
                        mask_path = '/media/nicolas/maso/Mario/mask/mask_%s.tif'%(codigo)
                        mask_map = wmf.read_map_raster(mask_path)
                        mask_vect = cu.Transform_Map2Basin(mask_map[0],mask_map[1])
                    else:
                        mask_vect = None
                    if mask_vect is not None:
                        if len(pos) == 0: # si no hay nc en ese paso de tiempo.
                            mean.append(np.nan)
                        else:
                            try:
                                mean.append(np.sum(mask_vect*rvec)/np.sum(mask_vect))
                            except: # para las que no hay mascara.
                                mean.append(np.nan)
                # se actualiza la media de todas las mascaras en el df.
                df.loc[dates.strftime('%Y-%m-%d %H:%M:%S')]=mean             
            else:
                pass

            #guarda binario y df, si guardar binaria paso a paso no me interesa rvecaccum
            if save_bin == True and len(codigos)==1 and path_res is not None:
                mean = []
                #guarda en binario 
                dentro = cu.rain_radar2basin_from_array(vec = rvec,
                    ruta_out = path_res,
                    fecha = dates,
                    dt = Dt,
                    umbral = umbral)

                #si guarda nc de ese timestep guarda clasificados
                if dentro == 0: 
                    hagalo = True
                else:
                    hagalo = False
                #mira si guarda o no los clasificados
                if save_class:
                    #Escribe el binario convectivo
                    aa = cuConv.rain_radar2basin_from_array(vec = rConv,
                        ruta_out = path_res+'_conv',
                        fecha = dates,
                        dt = Dt,
                        doit = hagalo)
                    #Escribe el binario estratiforme
                    aa = cuStra.rain_radar2basin_from_array(vec = rStra,
                        ruta_out = path_res+'_stra',
                        fecha = dates,
                        dt = Dt,
                        doit = hagalo)

                #guarda en df meanrainfall.
                try:
                    mean.append(rvec.mean())
                except:
                    mean.append(np.nan)
                df.loc[dates.strftime('%Y-%m-%d %H:%M:%S')]=mean

    if save_bin == True and len(codigos)==1 and path_res is not None:
        #Cierrra el binario y escribe encabezado
        cu.rain_radar2basin_from_array(status = 'close',ruta_out = path_res)
        print ('.bin & .hdr saved')
        if save_class:
            cuConv.rain_radar2basin_from_array(status = 'close',ruta_out = path_res+'_conv')
            cuStra.rain_radar2basin_from_array(status = 'close',ruta_out = path_res+'_stra')
            print ('.bin & .hdr escenarios saved')
    else:
        print ('.bin & .hdr NOT saved')

        #elige los retornos.
        if accum == True and path_tif is not None:
            cu.Transform_Basin2Map(rvec_accum,path_tif)
            return df,rvec_accum,dfaccum
        elif accum == True:
            return df,rvec_accum,dfaccum
        elif all_radextent:
            return df,radmatrix
        else:
            return df 

def radar_rain(start,end,rain_path,codefile,rainfile_name,nc_basin,ncells,Dt=300.0,ext='.hdr'):
    '''
    Reads rain fields (.bin or .hdr)
    Parameters
    ----------
    start        : initial date
    end          : final date
    Returns
    ----------
    pandas DataFrame or Series with mean radar rain
    
    Local version: changes for rain_path,info.nc_path for running with other basins and saving apart.
    '''
    
    start,end = pd.to_datetime(start),pd.to_datetime(end)
    file = check_rain_files(start,end,codefile,rain_path) #cambio de argumentos.
    
    #si ya existe el binario lo abre y saca el df.
    if file:
        file = rain_path+file
        if ext == '.hdr':
            obj =  hdr_to_series(file+'.hdr')
        else:
            obj =  bin_to_df(file,ncells)
        obj = obj.loc[start:end]
    #si no existe el binario
    else:
        codigos=[codefile]
        print ('WARNING: converting rain data, it may take a while')
        obj = get_radar_rain(start,end,Dt,nc_basin,codigos,
                             meanrain_ALL=False,save_bin=True,path_res=rain_path+rainfile_name,umbral=0.005,verbose=False)
        obj = obj.loc[start:end]
        
    return obj

def get_rainfall2sim(ConfigList,starts_m,end):
    #generacion o lectura de lluvia
    start_p,end_p = starts_m[-1],end
    start_p,end_p = (pd.to_datetime(start_p)- pd.Timedelta('5 min')),pd.to_datetime(end_p)
    #se leen rutas
    codefile = get_ruta(ConfigList,'name_proj')
    rain_path = get_ruta(ConfigList,'ruta_rain2')
    rainfile_name = file_format(start_p,end).split('-')[0]+'-'+file_format(start_p,end).split('-')[1]+'-'+codefile+'-sample_user'
    ruta_out_rain = rain_path+rainfile_name+'.bin'
    nc_basin = get_ruta(ConfigList,'ruta_nc2')
    cu = wmf.SimuBasin(rute=nc_basin)#,SimSlides=True)

    #lluvia
    rain_vect = radar_rain(start_p,end_p,rain_path,codefile,rainfile_name,nc_basin,cu.ncells,Dt=wmf.models.dt)
    return rain_vect,ruta_out_rain,cu

#---------------------------------------------
#---------------------------------------------
#Funciones de escritura de resultados - modelo
#---------------------------------------------
#---------------------------------------------

def write_pathsHists(rutaConfig,Qhist=True,Nhist=True,Shist=True,warming_steps=None):
    ''' #Genera archivos vacios para cada parametrizacion cuando no existe historia o si esta quiere renovarse. 
        Si se quiere dejar de crear rutas para alguno de los dos, se debe indicar False. e.g. Shist=False.
        Genera un dataframe con la primera fila de un qsim, ssim.hdr cualquiera, resultado de simulacion de la cuenca de
        interes; la ruta de este archivo debe estar indicado en el configfile.
        #Funcion no operacinal.
        #Argumentos:
        -rutaConfig: string, ruta del configfile.
        -Qhist: boolean, crear un Qhist. Default= True.
        -Shist: boolean, crear un Shist. Default= True.
    '''

    ListConfig = get_rutesList(rutaConfig)

    #Actualiza Qhist
    if Qhist:
        #se lee la ruta donde se va a escribir
        ruta_qhist = get_ruta(ListConfig,'ruta_qsim_hist')
        #se lee el archivo que se va a copiar
        ruta_qsim = get_ruta(ListConfig,'ruta_qsim_op')
        rutas_qsim=glob.glob(ruta_qsim+'*.csv')
        rutaqsim=rutas_qsim[0]
        #se leen los archivos de todas las par para crear Qhist para cada una.
        listrutas_qsim=np.sort(rutas_qsim)

        for i in listrutas_qsim:
            #Se lee el archivo Qsim de donde tomar la primera fila.
            qsim=pd.read_csv(rutaqsim,index_col=0,parse_dates=True)
            Qh=qsim#.iloc[[0]]
            nameQhistfile= i.split('/')[-1].split('_')[1][2:]
            #Pregunta si ya existe y si se quiere sobreescribir
            try:
                Lol = os.listdir(ruta_qhist.split('Q')[0])
                pos = Lol.index(ruta_qhist.split('/')[-1]+nameQhistfile)
                flag = raw_input('Aviso: El archivo Qhist: '+ruta_qhist+nameQhistfile+' ya existe, desea sobre-escribirlo, perdera la historia de este!! (S o N): ')
                if flag == 'S':
                    flag = True
                else:
                    flag = False
            except:
                flag = True
            #Guardado
            if flag:
                Qh.to_csv(ruta_qhist+nameQhistfile)
                print ('Aviso: Se crean '+nameQhistfile+', el archivo Qsim usado para crear las rutas es: '+rutaqsim)
            else:
                print ('Aviso: No se crean los Qhist')
    else: 
        print (' Aviso: No se crean archivos Qhist.')

    #Actualiza Shist
    if Shist:
        ruta_shist =  get_ruta(ListConfig,'ruta_MS_hist')
        ruta_ssim =  get_ruta(ListConfig,'ruta_sto_op')
        rutas_ssim=glob.glob(ruta_ssim+'*.StOhdr')
        rutassim=rutas_ssim[0]
        listrutas_ssim=np.sort(rutas_ssim)

        for j in listrutas_ssim:
            #Se lee el archivo Qsim de donde tomar la primera fila.
            ssim=pd.read_csv(j, header = 4, index_col = 5, parse_dates = True, usecols=(1,2,3,4,5,6))
            Sh=ssim.loc[ssim.index[warming_steps:]] # se descarta warming_steps
            nameShistfile=j.split('/')[-1].split('.')[0][6:]+'.csv'#+'hist.'+i.split('/')[-1].split('.')[-1]
            #Pregunta si ya existe y si se quiere sobreescribir
            try:
                Lold = os.listdir(ruta_shist.split('M')[0])
                pos = Lold.index(ruta_shist.split('/')[-1]+nameShistfile)
                flag = raw_input('Aviso: El archivo Shist: '+ruta_shist+nameShistfile+' ya existe, desea sobre-escribirlo, perdera la historia de este!! (S o N): ')
                if flag == 'S':
                    flag = True
                else:
                    flag = False
            except:
                flag = True
            #Guardado
            if flag:
                Sh.to_csv(ruta_shist+nameShistfile)
                print ('Aviso: Se crean los Shist , el archivo Ssim usado para crear las rutas es: '+rutassim)
            else:
                print ('Aviso: No se crean los Shist')
    else: 
        print (' Aviso: No se crean archivos S.')

    #Actualiza Nhist
    if Nhist:
        #se lee la ruta donde se va a escribir
        ruta_nhist = get_ruta(ListConfig,'ruta_nsim_hist')
        #se lee el archivo que se va a copiar
        ruta_nsim = get_ruta(ListConfig,'ruta_nsim_op')
        rutas_nsim=glob.glob(ruta_nsim+'*.csv')
        rutansim=rutas_nsim[0]
        #se leen los archivos de todas las par para crear Qhist para cada una.
        listrutas_nsim=np.sort(rutas_nsim)

        for i in listrutas_nsim:
            #Se lee el archivo Qsim de donde tomar la primera fila.
            nsim=pd.read_csv(rutansim,index_col=0,parse_dates=True)
            Nh=nsim#.iloc[[0]]
            nameNhistfile= i.split('/')[-1].split('_')[1][2:]
            #Pregunta si ya existe y si se quiere sobreescribir
            try:
                Lol = os.listdir(ruta_nhist.split('N')[0])
                pos = Lol.index(ruta_nhist.split('/')[-1]+nameNhistfile)
                flag = raw_input('Aviso: El archivo Nhist: '+ruta_nhist+nameNhistfile+' ya existe, desea sobre-escribirlo, perdera la historia de este!! (S o N): ')
                if flag == 'S':
                    flag = True
                else:
                    flag = False
            except:
                flag = True
            #Guardado
            if flag:
                Nh.to_csv(ruta_nhist+nameNhistfile)
                print ('Aviso: Se crean '+nameNhistfile+', el archivo Qsim usado para crear las rutas es: '+rutansim)
            else:
                print ('Aviso: No se crean los Nhist')
    else: 
        print (' Aviso: No se crean archivos Nhist.')



def Write_SlidesResults(ruta_slides_hdr,ListEjecs,R,ruta_out_slides,cu):
    ''' Escribe .bin y .hdr de resultados de simulacion de deslizamientos.
    ''' 
    #Archivo plano que dice cuales son las param que simularon deslizamientos 
    f = open(ruta_slides_hdr,'w')
    f.write('## Parametrizaciones Con Simulacion de Deslizamientos \n')
    f.write('Parametrizacion \t N_celdas_Desliza \n')
    #Termina de escribir el encabezado y escribe el binario.
    rec = 0
    for c,i in enumerate(ListEjecs):
        #Determina la cantidad de celdas que se deslizaron
        Slides = np.copy(R[c][0]['Slides_Map'])
        Nceldas_desliz = Slides[Slides!=0].shape[0]
        f.write('%s \t %d \n' % (i[3]+i[1], Nceldas_desliz))
        #si esta verbose dice lo que pasa 
        print ('Param '+i[3]+i[1]+' tiene '+str(Nceldas_desliz)+' celdas deslizadas.')
        #Escribe en el binario 
        rec = rec+1
        wmf.models.write_int_basin(ruta_out_slides, R[c][0]['Slides_Map'],rec,cu.ncells,1)
    f.close()
    
def write_qhist(rutaQsim,rutaQhist):
    ''' #Actualiza el archivo Qsimhist con el dataframe de la ultima ejecucion de la parametrizacion definida en las rutas de
        entrada. Abre el archivo Qhist y Qsim y hace append del ultimo dato, usa df.reindex(...,freq='5min') para que la
        historia quede organizada cronologicamente, por eso solo funciona si se ejecuta con una frecuencia con minuto = 5.
        La actualizacion de qhist siempre va un paso atras que la de 
        porque el archivo Qsim que genera la ejecucion
        empieza con un paso atras del que corre y el archivo Ssmim no, la actualizacion toma esa primera pos.
        #Funcion operacional.
        #Argumentos:
        -rutaQsim= string, ruta del Qsim
        -rutaQhist= string, ruta del Qhis
    '''
    ##Se actualizan los historicos de Qsim de la parametrizacion asociada.
    #Lee el almacenamiento actual
    Qactual = pd.read_csv(rutaQsim,index_col=0,parse_dates=True)
    #Lee el historico
    Qhist = pd.read_csv(rutaQhist,index_col=0,parse_dates=True)
    #Actualiza Qhist con Qactual.
    try:
        Qhist=Qhist.append(Qactual)#.iloc[[0]])#.sort_index(axis=1))
        #borra index repetidos, si los hay - la idea es que no haya pero si los hay no funciona el df.reindex
        # Qhist=Qhist.drop_duplicates()
        Qhist[Qhist.index.duplicated(keep='last')]=np.NaN
        Qhist = Qhist.dropna(how='all')
        #Guarda el archivo historico 
        Qhist.to_csv(rutaQhist)
        # Aviso
        print ('Aviso: Se ha actualizado el archivo de Qsim_historicos de: '+rutaQhist)
    except:
        print ('Aviso: no se esta actualizando Qhist en: '+rutaQhist)

def write_Stohist(ruta_Ssim,ruta_Shist,warming_steps):
    ''' #Actualiza el Ssimhist con el estado promedio de C.I. de cada tanque, copiandolo desde el .StOhdr a un json antes
        creado.Abre el archivo Shist y Ssim y hace append del ultimo dato, usa df.reindex(...,freq='5min') para que la
        historia quede organizada cronologicamente, por eso solo funciona si se ejecuta con una frecuencia con minuto = 5.
        #Funcion operacional.
        -ruta_Ssim: string, ruta  del archivo de condiciones antecedentes usada para escribir resultado de la simulacion.
        -ruta_Shist: string, ruta .json del archivo Shist.
    '''
    #Lee el almacenamiento actual
    Sactual = pd.read_csv(ruta_Ssim+'.StOhdr', header = 4, index_col = 5, parse_dates = True, usecols=(1,2,3,4,5,6))
    #Lee el historico
    Shist = pd.read_csv(ruta_Shist,index_col=0,parse_dates=True)
    #Actualiza
    Shist=Shist.append(Sactual.loc[Sactual.index[warming_steps:]])
    #borra index repetidos, si los hay - la idea es que no haya pero si los hay no funciona el df.reindex
    Shist[Shist.index.duplicated(keep='last')]=np.NaN
    Shist = Shist.dropna(how='all')

    #guarda el archivo
    Shist.to_csv(ruta_Shist)
    print ('Aviso: Se ha actualizado el archivo de Ssim_historicos de: '+ruta_Shist)

# write json of ci.
    
# CI={'0':0.8,
#     '1':0.1,
#     '2':5,
#     '3':10,
#     '4':0.1}

# path_sto='/media/nicolas/maso/Soraya/SHOp_files/LaPresidenta12m_Op/CI_beforeEv.json'

# with open(path_sto, 'w') as fp:
#     json.dump(CI, fp)

# with open(path_sto, 'r') as fp:
#     data = json.load(fp)
    
#-----------------------------------
#-----------------------------------
#Funciones de ejecucion modelo
#-----------------------------------
#-----------------------------------

def get_executionlists(ConfigList,ruta_out_rain,cu,starts_m,end,windows,warming_steps=48):
    
    #ruta inputs (configfile)
    DicCI=get_ConfigLines(ConfigList,'-CIr','Paths')
    DicPars=get_ConfigLines(ConfigList,'-c','Pars')

    #rutas denpasos salida (configfile)
    ruta_StoOp = get_ruta(ConfigList,'ruta_sto_op')
    ruta_QsimOp = get_ruta(ConfigList,'ruta_qsim_op')
    ruta_QsimH = get_ruta(ConfigList,'ruta_qsim_hist')
    ruta_MS_H = get_ruta(ConfigList,'ruta_MS_hist')
    ruta_out_slides = get_ruta(ConfigList, 'ruta_slides_op')
    ruta_slides_bin, ruta_slides_hdr = wmf.__Add_hdr_bin_2route__(ruta_out_slides)
    pm = wmf.read_mean_rain(ruta_out_rain.split('.')[0]+'.hdr')

    #Prepara las listas para setear las configuraciones
    ListEjecs = []

    for window,start_m in zip(windows,starts_m): 
        pos_start = pm.index.get_loc(start_m) #;print pos_start
        npasos = int((end-start_m).total_seconds()/wmf.models.dt) #+1 # siempre queda faltanfo un paso de 5m.. no se porque
        STARTid = window #;print STARTid
        #pars
        for PARid in np.sort(DicPars.keys()):
            #CIs
            for CIid in np.sort(DicCI.keys()):
                with open(DicCI[CIid], 'r') as f:
                    CI_dic = json.load(f)
                ListEjecs.append([cu, CIid, CI_dic, ruta_out_rain, PARid, DicPars[PARid], npasos, pos_start, STARTid, ruta_StoOp+PARid+CIid+'-'+STARTid, ruta_QsimOp+PARid+CIid+'-'+STARTid+'.csv', ruta_QsimH+PARid+CIid+'-'+STARTid+'.csv', ruta_MS_H+PARid+CIid+'-'+STARTid+'.csv',warming_steps])
        
    return ListEjecs

def get_qsim(ListEjecs,save_hist=True,verbose = True):
    '''
    Nota: falta agregar la parte de guardar MS en las pos de las estaciones de humedad.
    '''
    for L in ListEjecs:
        #read nc_basin, CI, par,  - start,end in rainfall
        cu=L[0]
        cu.set_Storage(L[2]['0'], 0)
        cu.set_Storage(L[2]['1'], 1)
        cu.set_Storage(L[2]['2'], 2)
        cu.set_Storage(L[2]['3'], 3)
        cu.set_Storage(L[2]['4'], 4)

        res = cu.run_shia(L[5],L[3],L[6],L[7], # si lo corro con el shape completo falla el seteo del rain.indx dentro de wmf
                         ruta_storage=L[9], kinematicN=12) # se guardan condiciones para la sgte corrida.

        #save df_simresults
        #operational qsim - without warming steps
        res[1].loc[res[1].index[L[13]:]].to_csv(L[10]) #####
        # saving historical data
        if save_hist:
            write_qhist(L[10],L[11])
            write_Stohist(L[9],L[12],L[13])
        if verbose:
            print ('Config. '+L[4]+L[1]+' ejecutado')

    return res


#--------------------------------------------
#--------------------------------------------
#Funciones de obtencion de resultados finales
#--------------------------------------------
#--------------------------------------------

def Qsim2Nsim(est,Qsim,path_curvascalsim,path_curvascalob):
    '''
    ----------
    Parameters:
    ----------
    Returns:
    ----------
    Notes:
    '''
    
    df_curvascalsim = pd.read_csv(path_curvascalsim,index_col=0)
    fc = df_curvascalsim.loc[est]
    
    df_curvascalob = pd.read_csv(path_curvascalob,index_col=0)
    

    dfNsim = pd.DataFrame([fc.c_up * (Qsim ** fc.alpha_up),
                 fc.c_down * (Qsim ** fc.alpha_down)]).T
    dfNsim.columns = ['ccs_up','ccs_down']
    
    if df_curvascalob['eq_type'].loc[est] == 1: 
        fc_o = df_curvascalob.loc[est]
        Nsim_cco =(Qsim/fc_o.c)**(1/fc_o.alpha)
        dfNsim['cco'] =  Nsim_cco *100. # cm  
    elif df_curvascalob['eq_type'].loc[est] == 2:
        fc_o = df_curvascalob.loc[est]
        Nsim_cco =((Qsim/fc_o.c1)**(1/fc_o.alpha1)) + fc_o.c2
        dfNsim['cco'] =  Nsim_cco *100. # cm   
    
    return dfNsim

def NiorQ(est,nobs= None,start=None,end=None,real_time=False,ni2q=True,path_curvascalob=None,path_evsH=None,csv=True):

    if nobs is None and start is not None and end is not None:
        if real_time:
            #NOBS
            selfn = cprv1.Nivel(codigo=est,passwd='12345',user='sora')
            
            nobs = selfn.level(start,end)
            nobs = nobs.resample('5T').mean()
            
        elif path_evsH is not None:
            if csv:
                path = path_evsH+str(est)+'_Hcorr.csv'
                N = pd.read_csv(path,index_col=0,parse_dates=True)
                N = N.resample('5T').mean()
                nobs = N[N.columns[0]][start:end]
            else:
                path = path_evsH+str(est)+'_Hcorr.json'
                N = pd.read_json(path)
                N = N.resample('5T').mean()
                nobs = N[N.columns[0]][start:end]
        else:
            print ('Warning: level is not defined.')
            
    if nobs is not None and ni2q == True and path_curvascalob is not None:
        df_curvascal = pd.read_csv(path_curvascalob,index_col=0)
        #que curva usar.
        if df_curvascal['eq_type'].loc[est] == 1:
            qobs = df_curvascal.loc[est].c*((nobs/100.)**df_curvascal.loc[est].alpha)
        else:
            qobs = df_curvascal.loc[est].c1*(((nobs/100.) + df_curvascal.loc[est].c2)**df_curvascal.loc[est].alpha1)
       #retornos
        return nobs,qobs
    else:
        print ('Warning: No return.')
        
def getNsims_4config(ConfigList,ListEjecs,ests,verbose = True,save_hist=True):
    '''
    Nota: Lee simulaciones, se pasa a Nsim, se arman los DFs para guardar.
    Posteriormente graficar y calcular performance
    
    '''
    #lectura de rutas
    path_curvascalob = get_ruta(ConfigList,'ruta_curvascalob')
    path_curvascalsim = get_ruta(ConfigList,'ruta_curvascalsim')
    path_nsim_op = get_ruta(ConfigList,'ruta_nsim_op')
    path_nsim_hist = get_ruta(ConfigList,'ruta_nsim_hist')
    # lee cuales son los tramos correspondientes
    dftramos =  pd.read_csv(get_ruta(ConfigList,'ruta_nc2_tramos'),index_col=0)
    #estaciones: para aquellas que hay curvacalsim
    df_curvascalsim = pd.read_csv(path_curvascalsim,index_col=0)
    # names nsim
    names_nsim = ['ccsup','ccsdown','cco']
    #holder de listas para graficar
#     list2plot
    
    for L in ListEjecs:
        #holders by config
        ccs_up = [] ; ccs_down = [] ; cco = []
        #names by config
        filenames_nsim = [path_nsim_op+'-' + namekey + L[10].split('/')[-1].split('_')[-1][2:] for namekey in names_nsim]
        filenames_nsimhist = [path_nsim_hist+'-' + namekey + L[10].split('/')[-1].split('_')[-1][2:] for namekey in names_nsim]
        #do
        dftramos =  pd.read_csv(get_ruta(ConfigList,'ruta_nc2_tramos'),index_col=0)
        qsim = pd.read_csv(L[10],index_col=0,parse_dates=True)
        dfqsim = qsim[map(str,(dftramos.loc[ests].values).T[0])]
        dfqsim.columns = ests

        for est in ests:
            nsim = Qsim2Nsim(est,dfqsim[est],path_curvascalsim,path_curvascalob)
        #     list2plot.append(nsim) ### esto debe es iterar por configuracion.
            ccs_up.append(nsim['ccs_up'])
            ccs_down.append(nsim['ccs_down'])
            cco.append(nsim['cco'])

        dfnsim_ccsup = pd.DataFrame(ccs_up).T
        dfnsim_ccsup.columns = ests
        dfnsim_ccsdown = pd.DataFrame(ccs_down).T
        dfnsim_ccsdown.columns = ests
        dfnsim_cco = pd.DataFrame(cco).T
        dfnsim_cco.columns = ests

        #save op. files
        dfnsim_ccsup.to_csv(filenames_nsim[0])
        dfnsim_ccsdown.to_csv(filenames_nsim[1])
        dfnsim_cco.to_csv(filenames_nsim[2])
        if verbose:
            print ('Aviso: dfNsim guardado en:')
            print (filenames_nsim[0])
            print (filenames_nsim[1])
            print (filenames_nsim[2])

        #save hist. files   
        if save_hist:
            dfs_act = [dfnsim_ccsup,dfnsim_ccsdown,dfnsim_cco]
            print ('Aviso: dfNsim_hist actualizado en:')

            for path_h,df_act in zip(filenames_nsimhist,dfs_act):
                df0 = pd.read_csv(path_h,index_col=0,parse_dates=True)
                df0.columns = list(map(int,df0.columns))
                df0=df0.append(df_act)
                df0[df0.index.duplicated(keep='last')]=np.NaN
                df0 = df0.dropna(how='all')
                df0.to_csv(path_h)
                print (path_h)

def setfile_obs_op(ConfigList,flagfile_age,start,end,path_curvascalob,ests):
    path_nobs_op = get_ruta(ConfigList,'ruta_nobs_op')
    path_qobs_op = get_ruta(ConfigList,'ruta_qobs_op')

    nobs_s = [] ; qobs_s = [] ; ns_s = []

    #si el flag file es nuevo (5minutos), lo sobreescribe
    if flagfile_age < 10 and flagfile_age >= 5:
        
        #NOBS,QOBS
        for est in ests:
            #trae obs.
            nobs,qobs = NiorQ(est,start=start,end=end,real_time=True,ni2q=True,path_curvascalob=path_curvascalob)
            nobs_s.append(nobs);qobs_s.append(qobs)
#         print ('1')
        #save
        dfnobs = pd.DataFrame(nobs_s).T
        dfnobs.columns = ests
        dfnobs.to_csv(path_nobs_op)
        dfqobs = pd.DataFrame(qobs_s).T
        dfqobs.columns = ests
        dfqobs.to_csv(path_qobs_op)
        print ('Aviso: Se escriben nuevos archivos %s , %s'%(path_nobs_op,path_qobs_op))

    #si no lo completa
    elif flagfile_age >= 10:
        
        start,end = end - pd.Timedelta('10m'),end + pd.Timedelta('10m') ############# end

        for est in ests:
            #trae obs.
            nobs,qobs = NiorQ(est,start=start,end=end,real_time=True,ni2q=True,path_curvascalob=path_curvascalob)
            nobs_s.append(nobs);qobs_s.append(qobs)
        #save
        dfnobs = pd.DataFrame(nobs_s).T
        dfnobs.columns = ests
        dfqobs = pd.DataFrame(qobs_s).T
        dfqobs.columns = ests
        #pega la ultima parte consultada.
        dfnobs_back = pd.read_csv(path_nobs_op,index_col=0,parse_dates=True)
        dfnobs_back.columns = list(map(int,dfnobs_back.columns))
        dfnobs_back = dfnobs_back.append(dfnobs)
        dfnobs_back[dfnobs_back.index.duplicated(keep='last')]=np.NaN
        dfnobs_back = dfnobs_back.dropna(how='all')
        dfnobs_back.to_csv(path_nobs_op)

        dfqobs_back = pd.read_csv(path_qobs_op,index_col=0,parse_dates=True)
        dfqobs_back.columns = list(map(int,dfqobs_back.columns))
        dfqobs_back = dfnobs_back.append(dfqobs)
        dfqobs_back[dfqobs_back.index.duplicated(keep='last')]=np.NaN
        dfqobs_back = dfqobs_back.dropna(how='all')
        dfqobs_back.to_csv(path_qobs_op)
        
        print ('Aviso: Se actualizan archivos %s , %s'%(path_nobs_op,path_qobs_op))
#     return dfqobs,dfnobs,path_qobs_op,path_no
                
def get_performance_op(ConfigList,ests,ListEjecs,set_file=1):
    '''
    Calcula performance de Qs y Ns de los archivos escritos.
    Nota:
    - set_file: 0 for write from scratch, 1 for updating file.
    '''
    dftramos =  pd.read_csv(get_ruta(ConfigList,'ruta_nc2_tramos'),index_col=0)
    # leer rutas de entrada
    path_nobs_op = get_ruta(ConfigList,'ruta_nobs_op')
    path_qobs_op = get_ruta(ConfigList,'ruta_qobs_op')
    path_nsim_op = get_ruta(ConfigList,'ruta_nsim_op')
    names_nsims = ['ccsup','ccsdown','cco']

    #+L[10].split('/')[-1].split('_')[-1][2:]

    #leer rutas de salida
    varss = ['qsim','nsim']
    criterias = ['ns','dmp','dm']

    namesl = ['ruta_qsim_tp_hist']
    for v in varss:
        for cr in criterias:
            if v == 'nsim':
                for name_nsim in names_nsims:
                    namesl.append('ruta_'+v+'_'+cr+'_'+name_nsim+'_hist')
            else:
                namesl.append('ruta_'+v+'_'+cr+'_hist')

    # lectura de archivos y generacion de resultados
    dfqobs = pd.read_csv(path_qobs_op,index_col=0,parse_dates=True)
    dfqobs.columns = list(map(int,dfqobs.columns))
    dfnobs = pd.read_csv(path_nobs_op,index_col=0,parse_dates=True)
    dfnobs.columns = list(map(int,dfnobs.columns))

    for L in ListEjecs:

        qsim = pd.read_csv(L[10],index_col=0,parse_dates=True)
        dfqsim = qsim[map(str,(dftramos.loc[ests].values).T[0])]
        dfqsim.columns = ests
        
        filenames_nsims = [path_nsim_op+'-' + namekey + L[10].split('/')[-1].split('_')[-1][2:] for namekey in names_nsims]
        dfnsim_ccsup = pd.read_csv(filenames_nsims[0],index_col=0,parse_dates=True)
        dfnsim_ccsup.columns = list(map(int,dfnsim_ccsup.columns))
        dfnsim_ccsdown = pd.read_csv(filenames_nsims[1],index_col=0,parse_dates=True)
        dfnsim_ccsdown.columns = list(map(int,dfnsim_ccsdown.columns))
        dfnsim_cco = pd.read_csv(filenames_nsims[2],index_col=0,parse_dates=True)
        dfnsim_cco.columns = list(map(int,dfnsim_cco.columns))
        
        Qns_s = []; Qdqmp_s = [] ; Qdqm_s = [] ; Qtp_s = []
        Nns_ups = [] ; Ndnmp_ups = [] ; Ndnm_ups = []
        Nns_downs = [] ; Ndnmp_downs = [] ; Ndnm_downs = []
        Nns_ccos = [] ; Ndnmp_ccos = [] ; Ndnm_ccos = []

        for est in ests:
            # si no hay datos para la est --> np.nan
            if np.isnan(dfqobs.mean().loc[est]) == True or np.isnan(dfqsim.mean().loc[est]) ==  True: 
                Qns_s.append(np.nan)
                Qdqmp_s.append(np.nan)
                Qdqm_s.append(np.nan)
                Qtp_s.append(np.nan)
                Nns_ups.append(np.nan)
                Ndnmp_ups.append(np.nan)
                Ndnm_ups.append(np.nan)
                Nns_downs.append(np.nan)
                Ndnmp_downs.append(np.nan)
                Ndnm_downs.append(np.nan)
                Nns_ccos.append(np.nan)
                Ndnmp_ccos.append(np.nan)
                Ndnm_ccos.append(np.nan)
            else:
                Qns_s.append(wmf.__eval_nash__(dfqsim[est],dfqobs[est])) #Qns_s
                Qdqmp_s.append(wmf.__eval_nash__(dfqsim[est],dfqobs[est])*-1)
                Qdqm_s.append(dfqsim[est].max()-dfqobs[est].max())
                Qtp_s.append((dfqsim[est].argmax()-dfqobs[est].argmax()).total_seconds()/60)
                Nns_ups.append(wmf.__eval_nash__(dfnsim_ccsup[est],dfnobs[est]))
                Ndnmp_ups.append(wmf.__eval_nash__(dfnsim_ccsup[est],dfnobs[est])*-1)
                Ndnm_ups.append(dfnsim_ccsup[est].max()-dfnobs[est].max())
                Nns_downs.append(wmf.__eval_nash__(dfnsim_ccsup[est],dfnobs[est]))
                Ndnmp_downs.append(wmf.__eval_nash__(dfnsim_ccsup[est],dfnobs[est])*-1)
                Ndnm_downs.append(dfnsim_ccsup[est].max()-dfnobs[est].max())
                Nns_ccos.append(wmf.__eval_nash__(dfnsim_ccsup[est],dfnobs[est]))
                Ndnmp_ccos.append(wmf.__eval_nash__(dfnsim_ccsup[est],dfnobs[est])*-1)
                Ndnm_ccos.append(dfnsim_ccsup[est].max()-dfnobs[est].max())

        #saving performance hist
        print ('Aviso: se calcula performance: '+L[10].split('/')[-1].split('_')[-1][2:])
        holders = [Qtp_s,Qns_s,Qdqmp_s,Qdqm_s,Nns_ups,Nns_downs,Nns_ccos,Ndnmp_ups,Ndnmp_downs,Ndnmp_ccos,Ndnm_ups,Ndnm_downs,Ndnm_ccos]
        for l,namel in zip(holders,namesl):
            df = pd.DataFrame(l).T
            df.columns =  dfqsim.columns
            df.index = [dfqsim.index[-1]]
            if set_file == 0:
                df.to_csv(get_ruta(ConfigList,namel)+L[10].split('/')[-1].split('_')[-1][2:])
            elif set_file == 1:
                df0 = pd.read_csv(get_ruta(ConfigList,namel)+L[10].split('/')[-1].split('_')[-1][2:],index_col=0,parse_dates=True)
                df0.columns = list(map(int,df0.columns))
                df0=df0.append(df)
                df0[df0.index.duplicated(keep='last')]=np.NaN
                df0 = df0.dropna(how='all')
                df0.to_csv(get_ruta(ConfigList,namel)+L[10].split('/')[-1].split('_')[-1][2:])
    
#-------------------------------------
#-------------------------------------
#Funciones de despliegue de resultados
#-------------------------------------
#-------------------------------------
        
def plot_Nsim4configs(ConfigList,ListEjecs,colors,window,fc_pmean=12,
                   separate_extrapolwindow='30m',rutafig = None,):

    #lectura de cosas
    path_nsim_op = get_ruta(ConfigList,'ruta_nsim_op')

    Nobs = pd.read_csv(get_ruta(ConfigList,'ruta_nobs_op'),index_col=0,parse_dates=True)
    Nobs.columns = list(map(int,Nobs.columns))
    Qobs = pd.read_csv(get_ruta(ConfigList,'ruta_qobs_op'),index_col=0,parse_dates=True)
    Qobs.columns = list(map(int,Qobs.columns))

    Prad = wmf.read_mean_rain(ListEjecs[0][3].split('.')[0]+'.hdr')

    dftramos =  pd.read_csv(get_ruta(ConfigList,'ruta_nc2_tramos'),index_col=0)
    path_nsim_op = get_ruta(ConfigList,'ruta_nsim_op')
    names_nsims = ['ccsup','ccsdown','cco']
    # namescol = ['nsim_ccsup','nsim_ccsdown','nsim_cco']

    #LEER RESULTADOS TODAS LAS PAR
    for ind,est in enumerate(Nobs.columns):
        df2plot_est_pars = []
    #     legend_labels = []
        for L in ListEjecs:

            filenames_nsims = [path_nsim_op+'-' + namekey + L[10].split('/')[-1].split('_')[-1][2:] for namekey in names_nsims]
    #         legend_labels.append(L[10].split('/')[-1].split('_')[-1].split('.')[0][3:])

            dfnsim_ccsup = pd.read_csv(filenames_nsims[0],index_col=0,parse_dates=True)
            dfnsim_ccsup.columns = list(map(int,dfnsim_ccsup.columns))
            dfnsim_ccsdown = pd.read_csv(filenames_nsims[1],index_col=0,parse_dates=True)
            dfnsim_ccsdown.columns = list(map(int,dfnsim_ccsdown.columns))
            dfnsim_cco = pd.read_csv(filenames_nsims[2],index_col=0,parse_dates=True)
            dfnsim_cco.columns = list(map(int,dfnsim_cco.columns))

            namescol = ['nsim_ccsup','nsim_ccsdown',L[10].split('/')[-1].split('_')[-1].split('.')[0][3:]]
            #df para la est con pars.
            dfest_par = pd.DataFrame([dfnsim_ccsup[est],dfnsim_ccsdown[est],dfnsim_cco[est]]).T #dfqsim[est],
            dfest_par.columns =  namescol
            dfest_par['Pmean'] =  Prad 
            df2plot_est_pars.append(dfest_par)

        #GRAFICA
    #     legend_labels.append('Nobs')
        selfN =  cprv1.Nivel(user='sora',passwd='12345',codigo=est)
        Nobs2plot = Nobs[[est]].loc[df2plot_est_pars[0].index[-1] - pd.Timedelta(window):]
        Nobs2plot.columns = ['Nobs']


        # fig properties
        pl.rc('axes',labelcolor='#4f4f4f')
        pl.rc('axes',linewidth=1.25)
        pl.rc('axes',edgecolor='#4f4f4f')
        pl.rc('text',color= '#4f4f4f')
        pl.rc('text',color= '#4f4f4f')
        pl.rc('xtick',color='#4f4f4f')
        pl.rc('ytick',color='#4f4f4f')

        fig=pl.figure(figsize=(10,5),dpi=100)
        ax=fig.add_subplot(111)
        if separate_extrapolwindow is not None:
            for df2plot,color in zip(df2plot_est_pars,colors):
                df2plot = df2plot.loc[df2plot.index[-1] - pd.Timedelta(window):]
                df2plot[df2plot.columns[:-1]].loc[:df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow)].plot(ax=ax,color=color,lw=3.2,alpha=1,style=['--',':','-'])
                df2plot[df2plot.columns[:-1]].loc[df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow):].plot(ax=ax,color=color,lw=3.2,style=['--',':','-'],alpha=0.55,legend=False)

            Nobs2plot.loc[:df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow)].plot(ax=ax,color='k',lw=3.5,label='Nobs')

        else:
            Nobs2plot.plot(ax=ax,color='k',lw=3.5,label='Nobs')
            for df2plot,color in zip(df2plot_est_pars,colors):
                df2plot = df2plot.loc[df2plot.index[-1] - pd.Timedelta(window):]
                df2plot[df2plot.columns[:-1]].plot(ax=ax,color=color,lw=3.2,alpha=1,style=['--',':','-'])

        ax.set_title('Est. '+str(est)+' | ' + str(selfN.infost[selfN.infost.keys()[1]].sort_index().loc[est]+'\n'+ 
                     df2plot.index[0].strftime('%Y-%m-%d %H:%M')+' _ '+df2plot.index[-1].strftime('%Y-%m-%d %H:%M')),
                     fontsize=18.5)
        ax.tick_params(labelsize=14)
        ax.set_ylabel('Nivel $(cm)$',fontsize=18)
        ax.set_xlabel(u'Tiempo',fontsize=18)

        #second axis
        axAX=pl.gca()
        ax2=ax.twinx()
        ax2AX=pl.gca()
        if separate_extrapolwindow is not None:
            (df2plot[df2plot.columns[-1]][df2plot.index[0]:df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow)]*fc_pmean).plot.area(ax=ax2,lw=1.5,color=['#78c4d0'],stacked=False,label = u'Precip. Obs.') 
            (df2plot[df2plot.columns[-1]][df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow):df2plot.index[-1]]*fc_pmean).plot.area(ax=ax2,lw=3,color=['#aedae4'],stacked=False,style=':',label = u'Precip. Extrapol.') 
        else:
            (df2plot[df2plot.columns[-1]][df2plot.index[0]:df2plot.index[-1]]*fc_pmean).plot.area(ax=ax2,lw=1.5,color=['#78c4d0'],stacked=False,label = u'Precip. Obs.') 
        ##80c8d3
        ax2.set_ylim(0,ax2AX.get_ylim()[1]*4.5)
        ax2AX.set_ylim(ax2AX.get_ylim() [::-1]) 
        ax2.tick_params(axis='both',labelsize=14)
        ax2AX.tick_params(axis='both',labelsize=14)
        ax2.set_ylabel(u'Precipitación Prom. $(mm.{h}^{-1})$',fontsize=18)

        # nrisks
        riskcolor=['k','green','yellow','orange','r']    
        Qrisks=np.insert(selfN.infost[['n1','n2','n3','n4']].loc[est].values,0,0)
        if ax.get_yticks()[-1] > Qrisks[-1]:
            pass
        else:
            ax.set_ylim(0,Qrisks[-1]+10)

        #ancho barra nrisk e espacio al inicio de la serie
        total_minlength = (df2plot.index[-1] - df2plot.index[0]).total_seconds()/60 #ventana de tiempo en min.
        t0 = int(total_minlength*0.05) # el 5perc de la ventana de tiempo 
        t1 = int(total_minlength*0.015) # el 1.5perc
        t2 = int(total_minlength*0.06) # el 6perc

        for index,qrisk in enumerate(Qrisks[:-1]):    
            ax.axvspan(df2plot.index[0] - pd.Timedelta('%sm'%t0),df2plot.index[0] - pd.Timedelta('%sm'%t1),Qrisks[index]/ax.get_ylim()[1],Qrisks[index+1]/ax.get_ylim()[1],color=riskcolor[index+1])
        ax.set_xlim(df2plot.index[0]  - pd.Timedelta('%sm'%t2), df2plot.index[-1] + pd.Timedelta('0m'))

        if separate_extrapolwindow is not None:
            ax.axvline(df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow),ymax=Qrisks[-1]+10,color='#4f4f4f',ls='--',alpha=0.7,lw=1.75)


        #se escogen las lineas a mostrar
        ind_labels=[]
        for ind,i in enumerate(ax.get_legend_handles_labels()[1]):
            if i.startswith('p'):
                ind_labels.append(ind)       
        if separate_extrapolwindow is not None:
            ind_codes = ind_labels[::2]
        else:
            ind_codes = ind_labels
        ind_codes.append(np.where(np.array(ax.get_legend_handles_labels()[1]) == u'Nobs')[0][0])    
        #legend    
        legend_codes = [ax.get_legend_handles_labels()[0][i] for i in ind_codes]
        legend_labels = [ax.get_legend_handles_labels()[1][i] for i in ind_codes]
        leg1 = ax.legend(legend_codes,legend_labels,fontsize=15,bbox_to_anchor =(1.01,-0.2),ncol=5)
        leg2 = ax2.legend(fontsize=15,bbox_to_anchor =(1.01,-0.325),ncol=2)

        if rutafig is not None:
                    pl.savefig(rutafig+'nsim/'+str(est)+'/'+'Nsim_'+str(est)+'_'+Nobs2plot.index[0].strftime('%Y%m%d%H%M')+                     '_'+Nobs2plot.index[-1].strftime('%Y%m%d%H%M')+'.png',bbox_inches='tight',bbox_extra_artists=[leg1,leg2])
            
def plot_Qsim4configs(ConfigList,ListEjecs,colors,window,fc_pmean=12,
                   separate_extrapolwindow='30m',rutafig = None):

    #lectura de cosas
    path_nsim_op = get_ruta(ConfigList,'ruta_qsim_op')

    Qobs = pd.read_csv(get_ruta(ConfigList,'ruta_qobs_op'),index_col=0,parse_dates=True)
    Qobs.columns = list(map(int,Qobs.columns))

    Prad = wmf.read_mean_rain(ListEjecs[0][3].split('.')[0]+'.hdr')

    dftramos =  pd.read_csv(get_ruta(ConfigList,'ruta_nc2_tramos'),index_col=0)

    #LEER RESULTADOS TODAS LAS PAR
    dfs2plot_est = []

    for ind,est in enumerate(Qobs.columns):

        df2plot_est_pars = []
        legend_labels = []
        for L in ListEjecs:

            qsim = pd.read_csv(L[10],index_col=0,parse_dates=True)
            dfqsim = qsim[map(str,(dftramos.loc[Qobs.columns].values).T[0])]
            dfqsim.columns = Qobs.columns

            legend_labels.append(L[10].split('/')[-1].split('.')[0].split('_')[-1][3:])

            df2plot_est_pars.append(dfqsim[est])


        #df para la est con pars.
        df2plot = pd.DataFrame(df2plot_est_pars).T
        df2plot.columns = legend_labels
        df2plot['Pmean'] =  Prad 

        #GRAFICA
    #         legend_labels.append('Qobs')
        selfN =  cprv1.Nivel(user='sora',passwd='12345',codigo=est)
        Qobs2plot = Qobs[[est]].loc[df2plot.index[-1] - pd.Timedelta(window):]
        Qobs2plot.columns = ['Qobs']


        # fig properties
        pl.rc('axes',labelcolor='#4f4f4f')
        pl.rc('axes',linewidth=1.25)
        pl.rc('axes',edgecolor='#4f4f4f')
        pl.rc('text',color= '#4f4f4f')
        pl.rc('text',color= '#4f4f4f')
        pl.rc('xtick',color='#4f4f4f')
        pl.rc('ytick',color='#4f4f4f')

        fig=pl.figure(figsize=(10,5),dpi=100)
        ax=fig.add_subplot(111)
        if separate_extrapolwindow is not None:
            df2plot = df2plot.loc[df2plot.index[-1] - pd.Timedelta(window):]
            df2plot[df2plot.columns[:-1]].loc[:df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow)].plot(ax=ax,color=colors,lw=3.2,alpha=1)
            df2plot[df2plot.columns[:-1]].loc[df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow):].plot(ax=ax,color=colors,lw=3.2,style=[':'],alpha=0.55)

            Qobs2plot.loc[:df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow)].plot(ax=ax,color='k',lw=3.5,label=u'Qobs')
        else:
            Qobs2plot.plot(ax=ax,color='k',lw=3.5,label=u'Qobs')
            df2plot = df2plot.loc[df2plot.index[-1] - pd.Timedelta(window):]
            df2plot[df2plot.columns[:-1]].plot(ax=ax,color=colors,lw=3.2,alpha=1)

        ax.set_title('Est. '+str(est)+' | ' + str(selfN.infost[selfN.infost.keys()[1]].sort_index().loc[est]+'\n'+ 
                     df2plot.index[0].strftime('%Y-%m-%d %H:%M')+' _ '+df2plot.index[-1].strftime('%Y-%m-%d %H:%M')),
                     fontsize=18.5)
        ax.tick_params(labelsize=14)
        ax.set_ylabel('Caudal $(m³.{s}^{-1})$',fontsize=18)
        ax.set_xlabel(u'Tiempo',fontsize=18)

        #second axis
        axAX=pl.gca()
        ax2=ax.twinx()
        ax2AX=pl.gca()
        if separate_extrapolwindow is not None:
            (df2plot[df2plot.columns[-1]][df2plot.index[0]:df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow)]*fc_pmean).plot.area(ax=ax2,lw=1.5,color=['#78c4d0'],stacked=False,label = u'Precip. Obs.') 
            (df2plot[df2plot.columns[-1]][df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow):df2plot.index[-1]]*fc_pmean).plot.area(ax=ax2,lw=3,color=['#aedae4'],stacked=False,style=':',label = u'Precip. Extrapol.') 
        else:
            (df2plot[df2plot.columns[-1]][df2plot.index[0]:df2plot.index[-1]]*fc_pmean).plot.area(ax=ax2,lw=1.5,color=['#78c4d0'],stacked=False,label = u'Precip. Obs.') 
        ##80c8d3
        ax2.set_ylim(0,ax2AX.get_ylim()[1]*4.5)
        ax2AX.set_ylim(ax2AX.get_ylim() [::-1]) 
        ax2.tick_params(axis='both',labelsize=14)
        ax2AX.tick_params(axis='both',labelsize=14)
        ax2.set_ylabel(u'Precipitación Prom. $(mm.{h}^{-1})$',fontsize=18)
        #real-time mark
        if separate_extrapolwindow is not None:
            ax.axvline(df2plot.index[-1] - pd.Timedelta(separate_extrapolwindow),ymax=ax.get_yticks()[-1]+10,color='#4f4f4f',ls='--',alpha=0.7,lw=1.75)
        #ylim
        ax.set_ylim(0,ax.get_yticks()[-1]+20)

        #se escogen las lineas a mostrar
        ind_labels=[]
        for ind,i in enumerate(ax.get_legend_handles_labels()[1]):
            if i.startswith('p'):
                ind_labels.append(ind)       
        if separate_extrapolwindow is not None:
            ind_codes = ind_labels[:2]
        else:
            ind_codes = ind_labels
        ind_codes.append(np.where(np.array(ax.get_legend_handles_labels()[1]) == u'Qobs')[0][0])    
        #legend    
        legend_codes = [ax.get_legend_handles_labels()[0][i] for i in ind_codes]
        legend_labels = [ax.get_legend_handles_labels()[1][i] for i in ind_codes]
        leg1 = ax.legend(legend_codes,legend_labels,fontsize=15,bbox_to_anchor =(1.01,-0.2),ncol=5)
        leg2 = ax2.legend(fontsize=15,bbox_to_anchor =(1.01,-0.325),ncol=2)

        if rutafig is not None:
            pl.savefig(rutafig+'qsim/'+str(est)+'/'+'Qsim_'+str(est)+'_'+Qobs2plot.index[0].strftime('%Y%m%d%H%M')+'_'+Qobs2plot.index[-1].strftime('%Y%m%d%H%M')+'.png',bbox_inches='tight',bbox_extra_artists=[leg1,leg2])
    
    
#-----------------------------------
#-----------------------------------
#Funciones de ejecucion operacional 
#-----------------------------------
#-----------------------------------

def run_model_op(ConfigList,flagfile_age,date,rutafig=None):
    #model settings
    set_modelsettings(ConfigList)
    #dates
    starts,starts_m,end,windows = time_windows(date,warming_window='4h',windows = ['3h','6h','12h'])
    #rainfall
    rain_vect,ruta_out_rain,cu = get_rainfall2sim(ConfigList,starts_m,end)
    # set of executions
    ListEjecs =  get_executionlists(ConfigList,ruta_out_rain,cu,starts_m,end,windows,warming_steps=48)
    # execution
    lists = [ListEjecs[0],ListEjecs[-1]]
    print (dt.datetime.now())
    res = get_qsim(lists,save_hist=True,verbose = True) #######ListEjecs
    print (dt.datetime.now())

    #inputs needed for reading results
    path_curvascalsim = get_ruta(ConfigList,'ruta_curvascalsim')
    path_curvascalob = get_ruta(ConfigList,'ruta_curvascalob')
    df_curvascalsim = pd.read_csv(path_curvascalsim,index_col=0)
    #estaciones: para aquellas que hay curvacalsim
    ests = df_curvascalsim.index
    #convierte qsim from files into nsim.
    getNsims_4config(ConfigList,lists,ests,save_hist=True) #######ListEjecs
    # observations query
    setfile_obs_op(ConfigList,flagfile_age,starts[-1],end,path_curvascalob,ests)
    # performance assessment
    get_performance_op(ConfigList,ests,lists,set_file=1)

    #inputs needed for plots
    window='12h'
    colors=['#C7D15D','#3CB371', '#22467F']
    #plottin
    plot_Nsim4configs(ConfigList,lists,colors,window,separate_extrapolwindow='30m',rutafig=rutafig)
    plot_Qsim4configs(ConfigList,lists,colors,window,separate_extrapolwindow='30m',rutafig=rutafig)

    
#-----
# CRON
#-----

def model_trigger(configfile,rutafig):
    import os
    import datetime
    #lee la ruta.
    ConfigList= get_rutesList(configfile)
    path_op= get_ruta(ConfigList,'ruta_proj_op')

    if any(np.array(os.listdir(path_op)) == 'flag2run') == True:
        datenow = pd.to_datetime(dt.datetime.strftime(round_time(dt.datetime.now()), '%Y-%m-%d %H:%M'))
        # fecha real para pruebas.
        date2run = datenow - pd.Timedelta('77 days')

        #se evalua la edad del archivo
        file_date = round_time(dt.datetime.fromtimestamp(os.path.getctime(path_op+'flag2run')))
        file_age = np.abs((datenow - pd.to_datetime(file_date)).total_seconds()/60.0) # mins
        print 'Flag file exist %s mins ago, so run between with date: %s'%(file_age, date2run)
        #running between 12h back till 30m after date2run.
        run_model_op(ConfigList,file_age,date2run,rutafig=rutafig)
    else:
        print 'Flag file does not exist'
        pass
        
def assess_flagfile(dfacum_past,dfacum_ahead,df_ref):
    for ID in df_ref.dropna().sort_index().index: # to each station with a threshold assigned         
        #assess if 3hacum is >= threshold and 3hacum will increase in next 30m
        if (dfacum_past[ID][-1] >= df_ref['threshold'].loc[ID]) & (dfacum_ahead[ID][-1] > dfacum_past[ID][-1]):
            #look for the flag file
            if 'flag2run' in os.listdir(df_ref['proj_paths'].loc[ID]):
                #si existe no pasa nada
                datenow=pd.to_datetime(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M'))
                file_date= SHop.round_time(dt.datetime.fromtimestamp(os.path.getctime(df_ref['proj_paths'].loc[ID]+'flag2run')))
                file_age = np.abs((datenow - pd.to_datetime(file_date)).total_seconds()/60.0)
                print 'Flag file age: '+str(file_age)+' min.'
            else:
                #si no existe lo crea
                f = open(df_ref['proj_paths'].loc[ID]+'flag2run','w')
                f.close()        
                print df_ref['proj_paths'].loc[ID]+'flag2run' + ' now exists.'

        #if not, look for the flag file and asses its age in all df_ref['proj_paths'] if they exist.
        else:
            print '%s basin has not reached its mean rainfall threshold.'%(ID)
            # look for the flag file
            if 'flag2run' in os.listdir(df_ref['proj_paths'].loc[ID]):
                # if it exists, asses its age i n  mins
                datenow=pd.to_datetime(dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M'))
                file_date= SHop.round_time(dt.datetime.fromtimestamp(os.path.getctime(df_ref['proj_paths'].loc[ID]+'flag2run')))
                file_age = np.abs((datenow - pd.to_datetime(file_date)).total_seconds()/60.0)
                print 'Flag file age: '+str(file_age)+' min.'
                #if is older than 6h - 360min, remomve it
                if file_age > 360: 
                    os.remove (df_ref['proj_paths'].loc[df_ref.index[index]]+'flag2run')
                    print 'Flag file has been removed'
                #if it's younger, nothing happens
                else:
                    pass
                    print 'Flag file stays.'
            #if it doesn't exist, nothing happens
            else:
                print 'There is no flag file.'
                pass


#-----------------
# ejecucion vieja
#-----------------
# def run_model_op(ConfigList,start,end,tramo):

#     start,end = (pd.to_datetime(start)- pd.Timedelta('5 min')),pd.to_datetime(end)
#     #se leen rutas
#     codefile=get_ruta(ConfigList,'name_proj')
#     rain_path=get_ruta(ConfigList,'ruta_rain')
#     rainfile_name= file_format(start,end).split('-')[0]+'-'+file_format(start,end).split('-')[1]+'-'+codefile+'-sample_user'
#     #file_format(start,end)[:-15]+codefile+'-sample_user' # la idea es que el file format permita hacer esto tambien..
#     ruta_out_rain=rain_path+rainfile_name+'.bin'
#     nc_basin=get_ruta(ConfigList,'ruta_nc')
#     cu=wmf.SimuBasin(rute=nc_basin,SimSlides=True)
    
#     #se genera lluvia. 
#     rain_vect = radar_rain(start,end,rain_path,codefile,rainfile_name,nc_basin,cu.ncells,Dt=300.0)
# #     rain_vect = radar_rain_vect(start,end,rain_path,codefile,rainfile_name,nc_basin,cu.ncells)
#     #se corre el modelo
#     Model_Ejec(ruta_out_rain,cu,ConfigList,Npasos=rain_vect.shape[0])
#     #se grafica
#     path_fuentes='/media/nicolas/Home/Jupyter/Sebastian/AvenirLTStd-Book/AvenirLTStd-Book.otf'
#     ylabel='Caudal $[m^{3}.s^{-1}]$'
#     xlabel='Tiempo'
#     ylabelax=u'Precipitación $[mm.h^{-1}$]'
#     fontsizeylabel=14
#     fontsizexlabel=14
#     plotPerformance(ConfigList,ruta_out_rain,tramo,end,
#                         path_fuentes,ylabel,xlabel,ylabelax,fontsizeylabel,fontsizexlabel)

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


    
def model_warper(L,verbose=True):
    ''' # Ejecuta directamente el modelo hidrologico de forma operacional para cada paso del tiempo y la proxima media hora,
        para esto: lee el binario de lluvia actual y la extrapolacion, si se programa de tal manera actualiza las CI., corre 
        y genera un archivo con el dataframe de la simulacon de Q en cada nodo para cada paso de tiempo y los binarios del
        estado final de almacenamiento de los tanques (.StObin y .StOhdr) que leera en la proxima ejecucion. Ademas corre y
        genera archivos .bin y .hdr de las celdas deslizadas.
        # Funcion operacional que solo se ejecuta desde la funcion Model_Ejec de alarmas.py.
        # Argumentos:
        -L= lista con tantas listas dentro como numero de parametrizaciones se quieran correr. Dentro debe tener en orden:
        los argumentos de la funcion cu.run_shia(), el nombre de la par. y las rutas de los archivos Ssim y Shist.
        La funcion Model_Ejec se encarga de crear las listas con todo lo necesario para las ejecucion model_warper().
    '''
    cu=L[0]
    #CI
    cu.set_Storage(wmf.models.max_capilar*L[2]['0'], 0)
    cu.set_Storage(L[2]['1'], 1)
    cu.set_Storage(L[2]['2'], 2)
    cu.set_Storage(L[2]['3'], 3)
    cu.set_Storage(L[2]['4'], 4)
    #whereto- binop - t+1
    whereto=pd.Series(np.zeros(L[6]-2))
    whereto[1]=1
    #Ejecucion del modelo
    Res = cu.run_shia(L[4],L[5],L[6]-2,L[7], #aun no entendo lo del -1, revisar modelo.
        ruta_storage=L[8], kinematicN=12,WheretoStore=whereto)
    #Escribe resultados 
    Qsim = Res[1]
    Qsim.to_json(L[9])
    #Actualiza historico de caudales simulados de la par. asociada.
#   write_qhist(L[9],L[10])
    #Se actualizan los historicos de humedad de la parametrizacion asociada.
#   write_Stohist(L[8],L[11])
    #imprime que ya ejecuto
    if verbose:
        print ('Par '+L[3]+L[1]+' ejecutado')
    return Res


def Model_Ejec(ruta_out_rain,cuenca,ConfigList,Npasos):
    #cuenca
    cu = cuenca
    #lectura de cosas en cofigfile
    DicCI=get_ConfigLines(ConfigList,'-CIr','Paths')
    DicPars=get_ConfigLines(ConfigList,'-c','Pars')
    ruta_modelset = get_ruta(ConfigList,'ruta_modelset')
    ruta_StoOp = get_ruta(ConfigList,'ruta_sto_op')
    ruta_QsimOp = get_ruta(ConfigList,'ruta_qsim_op')
    ruta_QsimH = get_ruta(ConfigList,'ruta_qsim_hist')
    ruta_MS_H = get_ruta(ConfigList,'ruta_MS_hist')
    ruta_out_slides = get_ruta(ConfigList, 'ruta_slides_op')
    ruta_slides_bin, ruta_slides_hdr = wmf.__Add_hdr_bin_2route__(ruta_out_slides)

    #Prepara las listas con los parametros necesarios para la ejecuciones.
    ListEjecs = []
    for parID in np.sort(list(DicPars.keys())):
        for CIid in np.sort(list(DicCI.keys())):
            with open(DicCI[CIid], 'r') as f:
                CI = json.load(f)
            ListEjecs.append([cu, CIid, CI, parID, DicPars[parID], ruta_out_rain, Npasos, 1, ruta_StoOp+parID+CIid, ruta_QsimOp+parID+CIid+'.json', ruta_QsimH+parID+CIid+'.json', ruta_MS_H+parID+CIid+'.json'])

    # model settings  Json
    with open(ruta_modelset, 'r') as f:
        model_set = json.load(f)
    # Model set
    wmf.models.retorno = model_set['retorno']
    wmf.models.show_storage = model_set['show_storage']
    wmf.models.separate_fluxes = model_set['separate_fluxes']
    wmf.models.dt = model_set['dt']
    wmf.models.sim_slides = model_set['sim_slides']
    wmf.models.sl_fs = model_set['sl_fs']
    cu.set_Slides(wmf.models.sl_zs * model_set['factor_correctorSL'], 'Zs') #factor corrector 1.

    #Ejecucion
    # Cantidad de procesos 
    Nprocess = len(ListEjecs)
    if Nprocess > 15:
        Nprocess = int(Nprocess/1.2)
    #Ejecucion  en paralelo y guarda caudales 
    print ('Resumen Ejecucion Modelo')
    print ('\n')
    p = Pool(processes=Nprocess)
    R = p.map(model_warper, ListEjecs)
    time.sleep(100) 
    p.terminate()
#     p.close()
    p.join()

    # Write slides simulation results
    if model_set['sim_slides'] == 1:
        print ('\n')
        print ('Resumen Deslizamientos')
        Write_SlidesResults(ruta_slides_hdr,ListEjecs,R,ruta_out_slides,cu)
        
    return R
    
def plotPerformance(ConfigList,ruta_out_rain,tramo,end,
                    path_fuentes,ylabel,xlabel,ylabelax,fontsizeylabel,fontsizexlabel,rutafig=True):
    import matplotlib.font_manager as fm
    import glob
    S=wmf.read_mean_rain(ruta_out_rain[:-4]+'.hdr')[:-1]
    Qs_paths=np.sort(glob.glob(get_ruta(ConfigList,'ruta_qsim_op')+'*'))
    #properties
    fonttype = fm.FontProperties(fname=path_fuentes)
    pl.rc('axes',labelcolor='#4f4f4f')
    pl.rc('axes',linewidth=1.25)
    pl.rc('axes',edgecolor='#4f4f4f')
    pl.rc('text',color= '#4f4f4f')
    pl.rc('text',color= '#4f4f4f')
    pl.rc('xtick',color='#4f4f4f')
    pl.rc('ytick',color='#4f4f4f')
    fonttype.set_size(9.5)
    legendfont=fonttype.copy()
    legendfont.set_size(12)
    colors=['#C7D15D','#3CB371', '#22467F']

    fig  =pl.figure(dpi=120,facecolor='w')
    ax = fig.add_subplot(111)
    for index,qs_path in enumerate(Qs_paths):
        Qs=pd.read_json(qs_path)
        Qs[tramo].plot(ax=ax,color=colors[index],lw=3.5,ls='--',label=str(qs_path.split('/')[-1].split('.')[0][4:]))

    pl.yticks(fontproperties=fonttype)
    pl.xticks(fontproperties=fonttype)
    ax.set_ylabel(ylabel,fontproperties=fonttype,fontsize=fontsizeylabel)
    ax.set_xlabel(xlabel,fontproperties=fonttype,fontsize=fontsizeylabel)
    #risk bar.
    Qrisks = [float(i[:-1]) for i in get_line(ConfigList,'Qrisks')]
    riskcolor=['k','green','yellow','orange','r']    
    ax.set_ylim(ax.get_yticks()[0],Qrisks[-1])
    for index,qrisk in enumerate(Qrisks[:-1]):    
        ax.axvspan(Qs[tramo].index[0],Qs[tramo].index[2],Qrisks[index]/ax.get_yticks()[-1],Qrisks[index+1]/ax.get_yticks()[-1],color=riskcolor[index+1])

    
    axAX=pl.gca()
    ax2=ax.twinx()
    ax2AX=pl.gca()
    S.plot.area(ax=ax2,alpha=0.2,label=u'Precipitación')
    ax2.set_ylim(0,)
    ax2AX.set_ylim(ax2AX.get_ylim() [::-1])
    ax2.legend(loc=(0,-0.3),prop=legendfont)
    ax.legend(loc=(0,-0.6),prop=legendfont)

    pl.yticks(fontproperties=fonttype)
    ax2.set_ylabel(ylabelax,fontproperties=fonttype,fontsize=fontsizeylabel)
#     if rutafig:
#         pathfig=get_ruta(ConfigList,'ruta_qsim_png')
#         pl.savefig(pathfig+str(end)+'.png',bbox_inches='tight',dpi=120,facecolor='w')
#         print ('Figure in '+pathfig+str(end)+'.png')

#-------------------------------------------------------------------------------------------    
    
####BACK, por si las moscas.
# def radar_rain(self,start,end,rain_path,rainfile_name,codefile,nc_path,ncells,radar_path,ext='.hdr'):
#     '''
#     Reads rain fields (.bin or .hdr)
#     Parameters
#     ----------
#     start        : initial date
#     end          : final date
#     Returns
#     ----------
#     pandas DataFrame or Series with mean radar rain
    
#     Local version: changes for self.rain_path,self.info.nc_path for running with other basins and saving apart.
#     '''
#     start,end = pd.to_datetime(start),pd.to_datetime(end)
#     file = check_rain_files(self,start,end,codefile,rain_path) #cambio de argumentos.
#     if file:
#         file = rain_path+file
#         if ext == '.hdr':
#             obj =  hdr_to_series(file+'.hdr')
#         else:
#             obj =  bin_to_df(self,file,ncells)
#         obj = obj.loc[start:end]
#     else:
#         print ('WARNING: converting rain data, it may take a while')
#         delay = datetime.timedelta(hours=5)
#         kwargs =  {
#                     'start':start+delay,
#                     'end':end+delay,
#                     'cuenca':nc_path,
#                     'rutaNc':radar_path,
#                     'rutaRes':rain_path+rainfile_name,#self.file_format(start,end),
#                     'dt':300,
#                     'umbral': 0.005,
#                     'verbose':False,
#                     'super_verbose':True,
#                     'old':None,
#                     'save_class':None,
#                     'store_true':None,
#                     'save_escenarios':None,
#                     'store_true':None,
#                    }
#         self.get_radar_rain(**kwargs)
#         file = rain_path + check_rain_files(self,start,end,codefile,rain_path) # cambio de argumentos
#         if ext == '.hdr':
#             obj =  hdr_to_series(file+'.hdr')
#         else:
#             obj =  bin_to_df(self,file,ncells)
#         obj = obj.loc[start:end]
#     return obj


# def radar_rain_vect(self,start,end,rain_path,rainfile_name,codefile,nc_path,ncells,radar_path,path_res**kwargs):
#     '''
#     Reads rain fields (.bin)
#     Parameters
#     ----------
#     start        : initial date
#     end          : final date
#     Returns
#     ----------
#     pandas DataFrame with datetime index and basin radar fields
#     '''
#     return radar_rain(self,start,end,rain_path,rainfile_name,codefile,nc_path,ncells,radar_path,path_res,ext='.bin',**kwargs)
    