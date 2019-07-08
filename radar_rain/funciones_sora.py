#!/usr/bin/env python
import os 
import pandas as pd
from wmf import wmf
import numpy as np 
import glob 
import json
import MySQLdb
import csv
from datetime import timedelta
import datetime
import datetime as dt
import pickle
import matplotlib.dates as mdates
import textwrap

from mpl_toolkits.basemap import Basemap
import netCDF4

##### cambio de fuente
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager

font_dirs = ['/media/nicolas/Home/Jupyter/Sebastian/AvenirLTStd-Book/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

matplotlib.rcParams['font.family'] = 'Avenir LT Std'
matplotlib.rcParams['font.size'] = 18

import pylab as pl

# ------------------------
# Consultas generales a BD
#-------------------------

def getInfoestIndatos(est_codes,host,user,passwd,bd):
    '''Consulta datos de las estaciones a la tabla estaciones del servidor del SIATA.'''
    codeest=est_codes[0]
    # coneccion a bd con usuario operacional
    host   = host
    user   = user
    passwd = passwd
    bd     = bd
    #Consulta a tabla estaciones
    Estaciones="SELECT codigo,longitude,latitude,nombreestacion,fechainstalacion  FROM estaciones WHERE codigo=("+str(codeest)+")"
    dbconn = MySQLdb.connect(host, user,passwd,bd)
    db_cursor = dbconn.cursor()
    db_cursor.execute(Estaciones)
    result = np.array(db_cursor.fetchall())
    estaciones_datos_all=pd.DataFrame(result,columns=['codigo','longitud','latitud','nombreestacion','fechainstalacion'])


    for ind,est in enumerate(est_codes[1:]):
        try:
            # codigo de la estacion.
            codeest=est
            # coneccion a bd con usuario operacional
            host   = host
            user   = user
            passwd = passwd
            bd     = bd
            #Consulta a tabla estaciones
            Estaciones="SELECT codigo,longitude,latitude,nombreestacion,fechainstalacion  FROM estaciones WHERE codigo=("+str(codeest)+")"
            dbconn = MySQLdb.connect(host, user,passwd,bd)
            db_cursor = dbconn.cursor()
            db_cursor.execute(Estaciones)
            result = np.array(db_cursor.fetchall())
            #holding
            estaciones_datos=pd.DataFrame(result,columns=['codigo','longitud','latitud','nombreestacion','fechainstalacion'])
            estaciones_datos_all=estaciones_datos_all.append(estaciones_datos)
        except:
            pass
    estaciones_datos_all.index=estaciones_datos_all['codigo']
    estaciones_datos_all.index.name=''
    estaciones_datos_all=estaciones_datos_all.drop('codigo',axis=1)
    return estaciones_datos_all

#-----------------
# Manejo de series
#-----------------
def FindMax(Ni,umbral,horasAtras=12,BusquedaAdelante=36):
    '''Nota: Q debe ser un masked_array'''
    Q = np.ma.array(Ni.values, mask=Ni.values > 1000);fechas=Ni.index.to_pydatetime()
    pos=np.where(Q>umbral)[0]
    grupos=[];g=[];Qmax=[]
    #Encuentra el maximo de cada grupo
    for pant,pnext in zip(pos[:-1],pos[1:]):        
        if pant+1>=pnext and pant+BusquedaAdelante>=pnext:
            g.append(pant)
        else:
            if len(g)>0:
                PosMaxGrupo=np.argmax(Q[g])
                grupos.append(g[PosMaxGrupo])
                Qmax.append(np.max(Q[g]))
            g=[]
    #Pule el maximo por si hay noData
    for c,g in enumerate(grupos):
        if Q.mask[g-1]:
            grupos.pop(c)
    #Obtiene las fechas 12 horas atras 
    if type(fechas)==list:
        fechas=__np.array(fechas)
    FechasAtras=fechas[grupos]-dt.timedelta(hours=horasAtras)
    fechas=list(fechas)     
    posAtras=[fechas.index(i) for i in FechasAtras]
    
    
    return  pd.Series(Q[grupos], index=Ni.index[grupos])

def pdfcdf(serie,bins):
    hr,b = np.histogram(serie,bins)
    hr = hr.astype(float) / hr.sum()
    #hr[hr == 0] = np.nan
    hrc = hr.cumsum()
    b = (b[:-1] + b[1:])/2
    return b,hr,hrc

def hietogram_frombins(Gs_eventos,tipo,years,nameruta):
    ''' Genera una matriz con los hietogramas de los eventos.
        Si tipo <> 'all_cells' se hace con el promedio de las celdas con lluvia.
    '''
    hietogram_0=np.zeros((Gs_eventos.size,73)) #tamano 73 depende del size de pos,ojo.
    for ind,i in  enumerate(Gs_eventos.index):
        serie=[]
        ruta_bin=list(years[nameruta][years[0]==str(i.year)])[0]
        ruta_hdr=list(years[nameruta][years[0]==str(i.year)])[0][:-3]+'hdr'
        #acumulado.
        DictRain = wmf.read_rain_struct(ruta_hdr)
        R = DictRain[u' Record']
        pos=R[i-pd.Timedelta('3 hours'):i+pd.Timedelta('3 hours')].values
        Vsum = np.zeros(cu.ncells)
        for p in pos:
            #se acumula la lluvia de la cuenca
            v,r = wmf.models.read_int_basin(ruta_bin,p,cu.ncells)
            #correcciones de rutina al valor
            v = v.astype(float); v = v/1000.0;v[dicpos['posrare']]=0.0
            #promedio de todas las celdas
            if tipo == 'all_cells':
                serie.append(v.mean())
            #promedio de celdas con lluvia
            elif v.mean()==0.0:
                serie.append(0.0)
            else:     
                serie.append(v[v!=0].mean())
        #se guarda la serie
        hietogram_0[ind]=serie
    return hietogram_0

def Find_IniEv(serie,percentile):
    ''' Devuelve la posicion del inicio de la primera creciente. La entrada debe solo contener un evento.
    '''
    return np.where(np.diff(serie)>=np.percentile(np.diff(serie),percentile))[0][0]

def get_rutesDF(paths,keysI,keysF,columns,indexes):
    ''' It returns a df with rutes of interest.
        - Arguments 1, 2, 3 and 4 shall have same size of paths to revew.
        - 4 shall have same size of pathsinside, those inside have to be same size too.
        '''
        
    import os
    listas=[]
    for index,path in enumerate(paths):
        lista=[]
        for pathinside in list(np.sort(os.listdir(path))):
                if pathinside.startswith(keysI[index]) and pathinside.endswith(keysF[index]):
                    lista.append((path+pathinside))
        listas.append(lista)
    df=pd.DataFrame(listas).T
    df.columns=columns
    df.index=indexes
    return df


# ------------------------------------------------------------------
#Los parametros fueron estimados con el trabajo del semestre pasado 
#parametros de recesion
a=0.0018911147247916646
b=1.6756286586629194
#parametros de BRM
f=0.19### el que mejir se ajusto a la mayor cantidad de hidrografas
k=0.009

def BRM(serie,f,k):#pd.Series
    Bt=[serie[0]]
    for i in range(len(serie)-1):
        if (serie[i+1] > Bt[-1] + k):
            Bt.append(Bt[-1]+k+f*(serie[i+1]-Bt[-1]))         
        else:
            Bt.append(serie[i+1])
    return pd.Series(Bt,index=serie.index)

#------------------
# Graficas
#------------------

def scatterplot(x,y,c,xlabel,ylabe,cbar_label,path_fuentes):
    #properties
    fonttype = fm.FontProperties(fname=path_fuentes)
    fonttype.set_size(16)
    legendfont=fonttype.copy()
    legendfont.set_size(15.5)

    fig = pl.figure(figsize=(8,15))
    fig.subplots_adjust(hspace=.35)
    ax1 = fig.add_subplot(211)
    pl.scatter(x, y, c = c, edgecolors='k',s = 150,
        cmap = pl.get_cmap('viridis'))
    ax1.plot([0,1],[1,0], 'b', lw = 0.8)
    ax1.plot([0,1],[0,1], 'k', lw = 0.8)
    #set del plot
    cbar1 = pl.colorbar()
    cbar1.ax.set_ylabel(cbar_label, size = 20,fontproperties=fonttype)
    for l in cbar1.ax.yaxis.get_ticklabels():
        l.set_font_properties(fonttype)
        l.set_size(15)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.grid(True)
    pl.yticks(fontproperties=fonttype)
    pl.xticks(fontproperties=fonttype)
    ax1.set_xlabel(xlabel, size = 20.5,fontproperties=fonttype)
    ax1.set_ylabel(ylabel, size = 20.5,fontproperties=fonttype)
    
def plotEventos(N,Evs,window,xlabel,ylabel,figsize):
    tm = pd.Timedelta(window)
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for p,i in enumerate(Evs.index):
        ax.plot(N[i - tm: i + tm].values)
    pl.grid()
    pl.xlabel(ylabel, size = 16)
    pl.ylabel(xlabel, size = 16)
    ax.tick_params(labelsize = 14)
    
    
#------------------------
#Paleta Nico
Paleta = {'c1':'#fde725',
    'c2':'#b5de2b',
    'c3':'#6ece58',
    'c4':'#35b779',
    'c5':'#1f9e89',
    'c6':'#26828e',
    'c7':'#2f658a',
    'c8':'#3e4989',
    'c9':'#482878'}

#--------------------
#paletas

def get_colors(cmapname='jet',listc_size=10):    
    from pylab import cm

    cmap = cm.get_cmap(cmapname, listc_size)    # PiYG

    colorbar=[]

    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        colorbar.append(matplotlib.colors.rgb2hex(rgb))
    return colorbar
#--------------------
    
##############################Funciones separa eventos####################
def FindMax(Ni,umbral,window2,horasAtras=12,BusquedaAdelante=36):
    '''Nota: Q debe ser un masked_array'''
    Q = np.ma.array(Ni.values, mask=Ni.values > 1000);fechas=Ni.index.to_pydatetime()
    pos=np.where(Q>umbral)[0]
    grupos=[];g=[];Qmax=[]
    #Encuentra el maximo de cada grupo
    for pant,pnext in zip(pos[:-1],pos[1:]):        
        if pant+1>=pnext and pant+BusquedaAdelante>=pnext:
            g.append(pant)
        else:
            if len(g)>0:
                PosMaxGrupo=np.argmax(Q[g])
                grupos.append(g[PosMaxGrupo])
                Qmax.append(np.max(Q[g]))
            g=[]
    #Pule el maximo por si hay noData
    for c,g in enumerate(grupos):
        if Q.mask[g-1]:
            grupos.pop(c)
    #Obtiene las fechas 12 horas atras 
    if type(fechas)==list:
        fechas=__np.array(fechas)
    FechasAtras=fechas[grupos]-dt.timedelta(hours=horasAtras)
    fechas=list(fechas)     
    posAtras=[fechas.index(i) for i in FechasAtras]

    #filtrar eventos repetidos.
    Evs = pd.Series(Q[grupos], index=Ni.index[grupos])
    deltaT=Evs.index[1:] - Evs.index[:-1]
    evsin2=np.where(deltaT >= window2)[0]
    
    return  Evs[evsin2]

def hydrographsfromSeries(Evs,Q,window):
    Qevs=[]
    for date in Evs.index:
        Qevs.append(Q[date - pd.Timedelta(window): date +  pd.Timedelta(window)].values)
    return pd.DataFrame(np.array(Qevs).T,columns=map(str,Evs.index))

def plotEventos(N,Evs,window,xlabel,ylabel,figsize):
    tm = pd.Timedelta(window)
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for p,i in enumerate(Evs.index):
        ax.plot(N[i - tm: i + tm].values)
    pl.grid()
    pl.xlabel(ylabel, size = 16)
    pl.ylabel(xlabel, size = 16)
    ax.tick_params(labelsize = 14)

def getEventsN(self,ni_h,Ests,umb_evs,window1,window2,pathdf= None,figure=True):
    '''
        Genera dfs con los eventos de las estaciones en Ests que deben tener su historico guardado
        el path, el umbral para considerar el evento por cada estacion estan en umb_evs.
    '''
    # si los umbrales son algunos  de los nrisk de alerta temprana se consultan
    if (umb_evs ==  'n1').any() or (umb_evs ==  'n2').any():
        pos_n1=np.where(umb_evs == 'n1')[0]
        pos_n2=np.where(umb_evs == 'n2')[0]
        umb_evs[pos_n1]=self.infost['n1'].loc[Ests[pos_n1]].values 
        umb_evs[pos_n2]=self.infost['n2'].loc[Ests[pos_n2]].values
    # to float
    umb_evs=np.array(map(float,umb_evs))
    Evs_All=[]
    for est,umb in zip(Ests,umb_evs):
#         ni_h=pd.read_json(path+'corr/'+str(est)+'_Hcorr.json')
        ni_h=ni_h.resample('5T').mean()
        Evs=FindMax(ni_h,umb,window1)
        Evs_All.append(Evs)
        dfNevs=hydrographsfromSeries(Evs,ni_h,window2)
        if pathdf is not None:
            dfNevs.to_csv(pathdf+'dfNevs5m_'+str(est)+'.csv')
        if figure:
            plotEventos(ni_h,Evs,window2,'Nivel $[cm]$ - Est. '+str(est),'Tiempo',(12,6))
    return Evs_All,umb_evs
    
def get_rutesDF(paths,keysI,keysF,columns,indexes):
    ''' It returns a df with rutes of interest.
        - Arguments 1, 2, 3 and 4 shall have same size of paths to revew.
        - 4 shall have same size of pathsinside, those inside have to be same size too.
        '''
        
    import os
    listas=[]
    for index,path in enumerate(paths):
        lista=[]
        for pathinside in list(np.sort(os.listdir(path))):
                if pathinside.startswith(keysI[index]) and pathinside.endswith(keysF[index]):
                    lista.append((path+pathinside))
        listas.append(lista)
    df=pd.DataFrame(listas).T
    df.columns=columns
    df.index=indexes
    return df

def PmeanfromBinPaths(Rainpaths,column,indexes):
    ''' Return a concatenated pd.Series with all rutes records Mean Rainfall.
       - import wmf'''
    mean=[];mean_or=[]
    for i in indexes:
        ruta_hdr=Rainpaths[column][i][:-3]+'hdr'
        seriep=wmf.read_mean_rain(ruta_hdr)
        mean.append(seriep)
        mean_or.append(seriep[str(i)])
    #serie
    seriemean=pd.concat(mean)
    seriemean_or=pd.concat(mean_or)
    return mean,seriemean,mean_or,seriemean_or

def RainAcum_Evs(cu,eventos,Rainpaths,column,window,poshalo):
    
    mapacum=np.zeros((eventos.size,cu.ncells))    
    for ind,date in  enumerate(eventos.index[:]):
        #RADAR
        ruta_bin=Rainpaths[column][Rainpaths.index==str(date.year)][0]
        ruta_hdr=Rainpaths[column][Rainpaths.index==str(date.year)][0][:-3]+'hdr'
        #acumulado.
        DictRain = wmf.read_rain_struct(ruta_hdr)
        R = DictRain[u' Record']
        pos=R[date-pd.Timedelta(window):date+pd.Timedelta(window)].values
        pos = pos[pos !=1]
        Vsum = np.zeros(cu.ncells)
        for p in pos:
            #se acumula la lluvia de la cuenca
            v,r = wmf.models.read_int_basin(ruta_bin,p,cu.ncells)
            #correcciones de rutina al valor
            v = v.astype(float); v = v/1000.0 ;v[poshalo]=np.nan 
            Vsum+=v
        mapacum[ind]=Vsum

    return pd.DataFrame(mapacum.T,columns=map(str,eventos.index))

def hietogramfromHDR(Evs,window,Rainpaths,column,indexes):
    Pmean,Pmean_or = PmeanfromBinPaths(Rainpaths,column,indexes)
    hieto=[]
    for ev in Evs.index:
        hieto.append(Pmean[ev-pd.Timedelta(window):ev+pd.Timedelta(window)])
    #hietograms
    return pd.DataFrame(np.array(hieto).T,columns=map(str,Evs.index))

def hietograms_filtered(Evs,window,Pmean):
    
    #bad dates between 2013-2017
    index_out=[]
    index_out.append(PmeanR['2013-02-01':'2013-02-11 10:00'])
    index_out.append(PmeanR['2013-06-13':'2013-06-18'])
    index_out.append(PmeanR['2013-09-13':'2013-10-22'])
    index_out.append(PmeanR['2013-12-15':'2013-12-26 12:00'])
    index_out.append(PmeanR['2014-09-18':'2014-10-10 12:00'])
    index_out.append(PmeanR['2016-11-06':'2016-12'])
    index_out.append(PmeanR['2017-02-24 12:00':'2017-02-26'])
    index_out.append(PmeanR['2017-11-07 01:00':'2017-11-11 12:00'])
    index_out.append(PmeanR['2017-11-20 03:00':'2017-11-29'])
    index_out = pd.concat(index_out)
    
    evs_inlist=[];evs_in=[]; evs_out=[]
    c=0
    for i in  Evs.index:
        ini=i-pd.Timedelta(window)
        fin=i+pd.Timedelta(window)
        #trata de encontrar el i:e del evento, si lo logra lo saca. Si no lo incluye.
        try:
            index_out.index.get_loc(ini)
            index_out.index.get_loc(fin)
            c+=1
            evs_out.append(i)
            pass
        except:
            evs_inlist.append(Pmean[ini:fin])
            evs_in.append(i)
    print (str(c) + 'events out!')
    #array fechas out, fechas in
    evs_out=np.array(evs_out); evs_in=np.array(evs_in)
    #dfeventos
    dfevs_in=pd.DataFrame(np.array(evs_inlist).T,columns=map(str,Evs[evs_in].index))
    return dfevs_in, evs_in, evs_out, index_out

def plotEv_Pacum_hydrograph(MapAcumDf,dfQevs,dfPevs,i,e,vmin=0,vmax=80.0): # cambiar paraque acum y plot aparezca al lado.
    indices=np.arange(MapAcumDf.keys().size)
    for index,key in zip(indices[i:e],MapAcumDf.keys()[i:e]):
        cu.Plot_basin(MapAcumDf[key],lines_spaces=0.08, vmin=vmin, vmax=vmax,
                     colorTable=pl.get_cmap('Spectral_r'),per_color='k',
                     colorbarLabel= 'Ev_'+str(index)+' '+str(key)+'\n Accumulated Rainfall [mm]')

        pl.figure()
        pl.title('Ev_'+str(index)+' '+str(key))
        ax= dfQevs[key].plot(c='k')
        ax.set_ylabel('Streamflow $[m^{3}.s^{-1}]$',fontsize=15)
        axAX=pl.gca()
        ax2=ax.twinx()
        ax2AX=pl.gca()
        dfPevs[key].plot.area(ax=ax2,alpha=0.25)
        ax2.set_ylabel('Mean Rainfall $[mm.h^{-1}]$',fontsize=15)
        ax2.set_ylim(0,)
        ax2AX.set_ylim(ax2AX.get_ylim() [::-1])       
        
################################################################RADAR.###############################


def get_radar_rain(start,end,Dt,cuenca,codigos,accum=False,path_tif=None,all_radextent=False,meanrain_ALL=True,complete_naninaccum=False,save_bin=False,save_class = False,path_res=None,umbral=0.005,rutaNC='/media/nicolas/Home/nicolas/101_RadarClass/',verbose=True):

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
    - df,rvecacum,dfacum if accum = True.
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
    
def acum_radarbasin(path_radar,cu, start=None,end=None, pos_point_hietogram=None,read_class=False):
    '''
    -----------
    Parameters:
    
     - pos_point_hietogram = Optional, default None. Dataframe with shape = (# pos_points,1)
    -----------
    Return:
    '''
    ruta_bin = path_radar+'.bin'
    ruta_hdr = path_radar+'.hdr'

    DictRain = wmf.read_rain_struct(ruta_hdr)
    R = DictRain[u' Record']
    Vsum = np.zeros(cu.ncells)
    
    if start is not None:
        pos = R[start:end].values
    else:
        pos = R.values
    #pos dentro de la serie de tiempo
    if pos_point_hietogram is not None:
        point_hietograms = np.zeros((pos.size,pos_point_hietogram.shape[0]))
        if read_class:
            ruta_binconv,ruta_hdrconv = path_radar+'_conv.bin',path_radar+'_conv.hdr'
            ruta_binstra,ruta_hdrstra = path_radar+'_stra.bin',path_radar+'_stra.hdr'  
            point_hietograms_conv = np.zeros((pos.size,pos_point_hietogram.shape[0]))
            point_hietograms_stra = np.zeros((pos.size,pos_point_hietogram.shape[0]))
    else:
        pos = pos[pos !=1]
    #acumulado
    for ind,p in zip(np.arange(pos.size),pos):
        #se acumula la lluvia de la cuenca
        v,r = wmf.models.read_int_basin(ruta_bin,p,cu.ncells)
        #correcciones de rutina al valor
        v = v.astype(float); v = v/1000.0 #;v[poshalo]=np.nan
        Vsum+=v
        # si se define saca hietograma en pixeles definidos
        if pos_point_hietogram is not None:
            point_hietograms[ind] = v[np.concatenate(pos_point_hietogram.values)]
            if read_class:
                vconv,r = wmf.models.read_int_basin(ruta_binconv,p,cu.ncells)
                vconv = vconv.astype(float);vconv = vconv/1000.0
                vstra,r = wmf.models.read_int_basin(ruta_binstra,p,cu.ncells)
                vstra = vstra.astype(float);vstra = vstra/1000.0
                point_hietograms_conv[ind] = vconv[np.concatenate(pos_point_hietogram.values)] 
                point_hietograms_stra[ind] = vstra[np.concatenate(pos_point_hietogram.values)] 
    if pos_point_hietogram is not None and read_class == True:
        df = pd.DataFrame([point_hietograms.T[0],point_hietograms.T[1],
        point_hietograms_conv.T[0],point_hietograms_conv.T[1],
        point_hietograms_stra.T[0],point_hietograms_stra.T[1]]).T
        cols = list(map(str,pos_point_hietogram.index))
        cols_conv = list(map(lambda x : x+'_conv',cols))
        cols_stra = list(map(lambda x : x+'_stra',cols))
        columns = np.concatenate([cols,cols_conv,cols_stra])
        df.columns = np.concatenate([cols,cols_conv,cols_stra])
        df.index  = R[start:end].index
        return Vsum,df
    elif pos_point_hietogram is not None:
        return Vsum,pd.DataFrame(point_hietograms,index=R.index[pos],columns=list(map(str,pos_point_hietogram.index)))
    else:
        return Vsum


############################### plot radar - FROM CPR
def radar_cmap(window_t,idlcolors=False):
    '''
    Parameters
    ----------
   
    Returns
    ----------
   
    '''
    import matplotlib.colors as colors
        
    if idlcolors == False:
        bar_colors=[(255, 255, 255),(0, 255, 255), (0, 0, 255),(70, 220, 45),(44, 141, 29),(255,255,75),(255,142,0),(255,0,0),(128,0,128),(102,0,102),(255, 153, 255)]
        if pd.Timedelta(window_t) < pd.Timedelta('7 days'):
            #WEEKLY,3h,EVENT.
            lev = np.array([0.,1.,5.,10.,20.,30.,45.,60., 80., 100., 150.])
        if pd.Timedelta(window_t) > pd.Timedelta('7 days'):
            #MONTHLY
            lev = np.array([1.,5.0,10.0,15.,20.0,25.0,30.0,40.0,50.,60.,70.0])*10.
    
    #IDL
    elif pd.Timedelta(window_t) <= pd.Timedelta('3h'):
        #coor para python
        bar_colors = [[  0, 255, 255],[  0, 255, 255],[  0, 255, 255],[  0,   0, 255],[ 70, 220,  45],[ 44, 141,  29],[255, 255,  75],[255, 200,  50],[255, 142,   0],[255,   0,   0],[128,   0, 128],[255, 153, 255]]        
        #original de juli en idl
        #bar_colors = [[200, 200, 200],[  0,   0,   0],[  0, 255, 255],[  0,   0, 255],[ 70, 220,  45],[ 44, 141,  29],[255, 255,  75],[255, 200,  50],[255, 142,   0],[255,   0,   0],[128,   0, 128],[255, 153, 255]]
        #3h
        lev  = np.array([0.2,1,2,4,6,8,10,13,16,20,24,30])*5.
    elif pd.Timedelta(window_t) > pd.Timedelta('3h'):
        #coor para python
        bar_colors = [[  0, 255, 255],[  0, 255, 255],[  0, 255, 255],[  0,   0, 255],[ 70, 220,  45],[ 44, 141,  29],[255, 255,  75],[255, 200,  50],[255, 142,   0],[255,   0,   0],[128,   0, 128],[255, 153, 255]]        
        #original de juli en idl
        #bar_colors = [[200, 200, 200],[  0,   0,   0],[  0, 255, 255],[  0,   0, 255],[ 70, 220,  45],[ 44, 141,  29],[255, 255,  75],[255, 200,  50],[255, 142,   0],[255,   0,   0],[128,   0, 128],[255, 153, 255]]
        #WEEKLY
        lev = np.array([0.2,1,4,7,10,13,16,20,25,30,35,50])*10.
    
    scale_factor =  ((255-0.)/(lev.max() - lev.min()))
    new_Limits = list(np.array(np.round((lev-lev.min())*scale_factor/255.,3),dtype = float))
    Custom_Color = list(map(lambda x: tuple(ti/255. for ti in x) , bar_colors))
    nueva_tupla = [((new_Limits[i]),Custom_Color[i],) for i in range(len(Custom_Color))]
    cmap_radar =colors.LinearSegmentedColormap.from_list('RADAR',nueva_tupla)
    levels_nuevos = np.linspace(np.min(lev),np.max(lev),255)
    norm_new_radar = colors.BoundaryNorm(boundaries=levels_nuevos, ncolors=256)
    return cmap_radar,list(levels_nuevos),norm_new_radar

def longitude_latitude_basin(self):
    '''
    Gets last topo-batimetry in db
    Parameters
    ----------
    x_sensor   :   x location of sensor or point to adjust topo-batimetry
    Returns
    ----------
    last topo-batimetry in db, DataFrame
    '''
    mcols,mrows = wmf.cu.basin_2map_find(self.structure,self.ncells)
    mapa,mxll,myll=wmf.cu.basin_2map(self.structure,self.structure[0],mcols,mrows,self.ncells)
    longs = np.array([mxll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(mcols)])
    lats  = np.array([myll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(mrows)])
    return longs,lats

def adjust_basin(cu,rel=0.766,fac=0.0):
    '''

    Parameters
    ----------

    ----------

    '''
    longs,lats = longitude_latitude_basin(cu)
    x = longs[-1]-longs[0]
    y = lats[-1] - lats[0]
    if x>y:
        extra_long = 0
        extra_lat = (rel*x-y)/2.0
    else:
        extra_lat=0
        extra_long = (y/(2.0*rel))-(x/2.0)
    return extra_lat+fac,extra_long+fac

def basin_mappable(cu,vec=None, extra_long=0,extra_lat=0,perimeter_keys={},contour_keys={},**kwargs):
    '''
    Gets last topo-batimetry in db
    Parameters
    ----------
    x_sensor   :   x location of sensor or point to adjust topo-batimetry
    Returns
    ----------
    last topo-batimetry in db, DataFrame
    '''
    longs,lats=longitude_latitude_basin(cu)
    x,y=np.meshgrid(longs,lats)
    y=y[::-1]
    # map settings
    m = Basemap(projection='merc',llcrnrlat=lats.min()-extra_lat, urcrnrlat=lats.max()+extra_lat,
        llcrnrlon=longs.min()-extra_long, urcrnrlon=longs.max()+extra_long, resolution='c',**kwargs)
    # perimeter plot
    xp,yp = m(cu.Polygon[0], cu.Polygon[1])
    m.plot(xp, yp,**perimeter_keys)
    # vector plot
    if vec is not None:
        map_vec,mxll,myll=wmf.cu.basin_2map(cu.structure,vec,len(longs),len(lats),cu.ncells)
        map_vec[map_vec==wmf.cu.nodata]=np.nan
        xm,ym=m(x,y)
        contour = m.contourf(xm, ym, map_vec.T, 25,**contour_keys)
    else:
        contour = None
    return m,contour

def plot_basin_rain(cu,vec,codigo,window_t='5 days',idlcolors=False,cbar=None,ax=None,**kwargs):
    '''
    Gets last topo-batimetry in db
    Parameters
    ----------
    x_sensor   :   x location of sensor or point to adjust topo-batimetry
    Returns
    ----------
    last topo-batimetry in db, DataFrame
    '''
    if ax is None:
        fig = plt.figure(figsize=(10,16))
        ax = fig.add_subplot()
    cmap_radar,levels,norm = radar_cmap(window_t,idlcolors=idlcolors)
    extra_lat,extra_long = adjust_basin(cu,fac=0.01)
    mapa,contour = basin_mappable(cu,
                                  vec,
                                  ax=ax,
                                  extra_long=extra_long,
                                  extra_lat = extra_lat,
                                  contour_keys={'cmap'  :cmap_radar,
                                                'levels':levels,
                                                'norm'  :norm},
                                 perimeter_keys={'color':'k'})
    if cbar:
        cbar = mapa.colorbar(contour,location='right',pad="15%")
        cbar.ax.set_title('mm',fontsize=14)
    else:
        cbar = mapa.colorbar(contour,location='right',pad="15%")
        cbar.remove()
        plt.draw()
    try:
        net_path = kwargs.get('net_path',"/media/nicolas/maso/Mario/shapes/net/%s/%s"%(codigo,codigo))
        stream_path = kwargs.get('stream_path',"/media/nicolas/maso/Mario/shapes/stream/%s/%s"%(codigo,codigo))
        mapa.readshapefile(net_path,'net_path')
        mapa.readshapefile(stream_path,'stream_path',linewidth=1)
    except:
        pass
    return mapa

def plot_allradarextent(rad2plot,window_t,idlcolors=False,path_figure=None,extrapol_axislims=False):
    '''
    Plot the whole radar matrix for web page.
    Parameters:
    ----------
    - rad2plot:      matrix.
    - path_figure:   string, path whete to save figure. Default None.
    Returns:
    ----------
    - None
    '''
    cmap_radar,levels,norm = radar_cmap(window_t,idlcolors=idlcolors)
    rad2plot[rad2plot == 0 ]=np.nan

    #plot
    fig = pl.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ims = ax.imshow(rad2plot,cmap=cmap_radar)
    ims.axes.get_xaxis().set_visible(False)
    ims.axes.get_yaxis().set_visible(False)
    ax.set_axis_off()
    ax.set_ylabel('y')
    #obs.
    if extrapol_axislims:
        #extrapol
        ax.set_xlim(50,1580)
        ax.set_ylim(1580,120)
    else:
        #obs.
        ax.set_xlim(50,1580)
        ax.set_ylim(1585,145)

    if path_figure is not None:
        pl.savefig(path_figure,bbox_inches='tight',dpi=100, transparent=True)

def plot_extrapol(idlcolors=False):
    # inputs acumula radar

    Dt=300.
    nc_basin= '/media/nicolas/maso/Mario/basins/260.nc'
    codigos = [260]
    accum=False;path_tif=None;meanrain_ALL=True;save_bin=False;path_res=None,
    umbral=0.005;rutaNC='/media/nicolas/Home/nicolas/101_RadarClass/'
    path_figs= '/media/nicolas/Home/Jupyter/Soraya/Op_Alarmas/Result_to_web/operacional/acum_radar/'
    
    starts = [round_time(dt.datetime.now()) - pd.Timedelta('3h'),round_time(dt.datetime.now()) ]
    ends = [round_time(dt.datetime.now()), round_time(dt.datetime.now()) + pd.Timedelta('30m')]
    figsnames = ['30minbefore_allradarextent','30minahead_allradarextent']
    
    for start,end,figname in zip(starts,ends,figsnames):
        # Acumula radar.
        dflol,radmatrix = get_radar_rain(start,end,Dt,nc_basin,codigos,all_radextent=True)
        # inputs fig
        path_figure =  path_figs+figname+'.png'
        rad2plot = radmatrix.T
        window_t='30m'
        #fig
        plot_allradarextent(rad2plot,window_t,idlcolors=idlcolors,path_figure=path_figure,extrapol_axislims=True)
        
def plot_acum_radar(windows_t,cbar=False,title=False):
    
    #DEFINICION DE COSAS
    rutafig= '/media/nicolas/Home/Jupyter/Soraya/Op_Alarmas/Result_to_web/operacional/acum_radar/prueba/'
    selfN = cprv1.Nivel(codigo=260,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
    codigos= selfN.infost.index[:]
    #setting the est indexes and order.
    pos_1st= [51,36,37,40,54,64]
    codigos = np.delete(codigos,pos_1st)
    codigos = np.insert(codigos,0,260)

    #PLOTS
    for window_t in windows_t[:2]:
        if window_t == '30m_ahead':
            start = pd.to_datetime('2019-05-09 22:00')
#             start= fs.round_time(dt.datetime.now())
            end = start + pd.Timedelta('30m')
        else:
            end = pd.to_datetime('2019-05-09 22:00')
#             end = fs.round_time(dt.datetime.now())
            start= end -  pd.Timedelta(window_t)
    
        path_basins= '/media/nicolas/maso/Mario/basins/'
        
        #acumulado para la cuenca mas grande
        path_radtif = '/media/nicolas/maso/Soraya/op_files/radar/tifs/260-'+window_t+'.tif'
        #solo guarda promedios por cuenca en algunas ventanas de tiempo
        if window_t in ['30m_ahead','3h','24h']:
            dfAll,rvec = fs.get_radar_rain(start,end,300.,path_basins+'%s.nc'%(260),codigos,accum=True,
                                path_tif = path_radtif,#+start.strftime('%Y%m%d%H%M')+'_'+start.strftime('%Y%m%d%H%M')+'.tif',
                                meanrain_ALL=True)
            #guarda csv
            dfAll.to_csv(rutafig+window_t+'/dfAcum'+window_t+'.csv')
        else:
            dfAll,rvec = fs.get_radar_rain(start,end,300.,path_basins+'%s.nc'%(260),codigos,accum=True,
                                path_tif = path_radtif,
                                meanrain_ALL=False)
        
        #plots
        for codigo in codigos[0:]:
            if '%s.nc'%(codigo) in os.listdir(path_basins):
                cu = wmf.SimuBasin(rute=path_basins+'%s.nc'%(codigo))
                a,b = wmf.read_map_raster(path_radtif)
                vec_rain = cu.Transform_Map2Basin(a,b)
                #Plot
                fig = pl.figure(figsize=(10,12))
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                if window_t == '30m_ahead':
                    fs.plot_basin_rain(cu,vec_rain,codigo,window_t='30m',ax=ax,cbar=cbar)
                else:
                    fs.plot_basin_rain(cu,vec_rain,codigo,window_t=window_t,ax=ax,cbar=cbar)
                if title:
                    ax.set_title('Est. ' +str(codigo) +' | '+ selfN.infost.loc[codigo].nombre +'\n'+pd.to_datetime(start).strftime('%Y%m%d%H%M')+'-'+pd.to_datetime(end).strftime('%Y%m%d%H%M'))
                pl.savefig(rutafig+window_t+'/'+selfN.infost.loc[codigo].slug+'.png',bbox_inches='tight')
    
    return dfAll,rvec
        
####################### traza cuencas


from wmf import wmf
import numpy as np
import pylab as pl 
import pandas as pd
import os 
import glob
import matplotlib.pyplot as plt
import time
import datetime as dt


def set_ncSimubasin(rutaDEM,rutaDIR,lat1,lon1,lat2,lon2,deltaT,name,umbral_red,dxp,
                    noDataP,rutabasinSHP,rutanetSHP,rutamapZ,rutaTetas,rutamapKs,rutamapMan,
                    epsilonLad,e1Lad,CoefOCG_agr,SimSlides,rutamapAnguloFric,rutamapCohesion,
                    rutamapPesoEsp,speedtype,rutaNC,
                    xy=None,xy_edgecolor=None,xy_s=None,xy_lw=None,rutaShp=None,
                    ShpIsPolygon=None,shpColor=None,shpWidth=None,shpAlpha=None):
    ''' Traza, prepara y guarda NC para simulacion, setea por defecto SimSlides.
        Devuelve figuras de todos los parametros y diccionario con cu.GeoParameters.
        Permite setear coeficientes y exponentes de ecuaciones no lineales del Subsup. y Cauce.
        Ademas permite setear la version de las ecuaciones con la que se quiere guardar el .nc (speedtype)
        
        Guarda:
        - .nc
        - .shp's de cuenca y red de drenaje
        
        Retornos:
        - cu : Simunbasin instance, con todas las variables de estado cargadas,
                Disponible para revisar lo necesario para verificar la consistencia del montaje.
        - cu.GeoParameters : Diccionario con resumen de parametros morfologicos'''
    
    #Cargado de mapas de entrada
    DEM=wmf.read_map_raster(rutaDEM,isDEMorDIR=True, dxp=dxp, noDataP=noDataP)
    DIR=wmf.read_map_raster(rutaDIR,isDEMorDIR=True, dxp=dxp, noDataP=noDataP,
        isDIR = True)
    #stream, aguas abajo que Simubasin... un punto cualquiera para que permita trazar la cuenca. 
    st = wmf.Stream(lat1,lon1, DEM, DIR, name=name)
    #cuenca ------------------------------------# FALTA SETEAR PARA hills
    cu = wmf.SimuBasin(lat2,lon2,DEM,DIR,stream = st, umbral = umbral_red,name=name,
                    noData=noDataP, dt=deltaT)

    #----------Plot 1
    fig=pl.figure(dpi=100)
    cu.Plot_basin(  cu.CellAcum,
                    lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='CellAcum $[celdas]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw)
    


    #-----------Saving 1
    cu.Save_Basin2Map(rutabasinSHP+'.shp')
    cu.Save_Net2Map(rutanetSHP+'.shp')

    #MORFOLOGIA BASICA
    cu.GetGeo_Cell_Basics()
    cu.set_Geomorphology(stream_width=cu.CellLong)
    cu.GetGeo_Parameters()

    #HAND
    cu.GetGeo_HAND()
    #----------Plot 2
    fig=pl.figure(dpi=100)
    cu.Plot_basin(cu.CellHAND,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='HAND $[m.t.n.d]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)


    #------------------#GENERACION DE MAPAS

    # Evaporacin en la cuenca estimada por Turc (?)
    Evp=4.658*np.exp(-0.0002*cu.CellHeight)
    #----------Plot 3
    fig=pl.figure(dpi=100)
    cu.Plot_basin(Evp,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Evp - Turc $[mm/d]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)


    #Lectura de la profunidad de raiz
    Z,p = wmf.read_map_raster(rutamapZ)
    Z = cu.Transform_Map2Basin(Z,p)
    Z[Z == -9999] = Z.mean()
    Z[Z==0]=Z.mean()

    #----------Plot 4
    fig=pl.figure(dpi=100)
    cu.Plot_basin(Z,
                 lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Prof. suelo $[m]$ (POMCA,2007)',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    #Profundidad por geomorfologia (Osorio,2006)
    Zg = np.zeros(cu.ncells)
    Zg[cu.CellSlope<0.25]=0.6
    Zg[(cu.CellSlope>=0.25)&(cu.CellSlope<0.30)]=1.0
    Zg[(cu.CellSlope>=0.30)&(cu.CellSlope<0.50)]=0.3
    Zg[cu.CellSlope>=0.5] = 0.2

    #----------Plot 5
    fig=pl.figure(dpi=100)
    cu.Plot_basin(Zg,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Prof. Suelo  $[m]$ (Osorio,2006)',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    #Contenido de agua
    Tetas = {}
    for i in ['Teta_pmp','Teta_cp','Teta_sat']:
        te,p = wmf.read_map_raster(rutaTetas+i+'.tif')
        te = cu.Transform_Map2Basin(te,p)
        te[te == -9999] = te[te>0].mean()
        te[te == 0] = te[te>0].mean()
        Tetas.update({i:te})

    Hu  = Zg * (Tetas['Teta_cp']-Tetas['Teta_pmp'])*10
    Hg  = Zg * (Tetas['Teta_sat']-Tetas['Teta_cp'])*10
    # sobre la zona urbana, se setean el almacenamineto capilar en la capacidad minima 
    Hu[Z==2]=Hu.min()

    #----------Plot 6

    fig=pl.figure(dpi=100)
    cu.Plot_basin(Hu,
                 lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Alm. Capilar $Hu$ $[mm]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    #----------Plot 7

    fig=pl.figure(dpi=100)
    cu.Plot_basin(Hg,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Alm. Gravitacional $Hg$ $[mm]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    Ks, p = wmf.read_map_raster(rutamapKs)
    Ks = cu.Transform_Map2Basin(Ks,p)
    Ks[Ks == -9999] = Ks[Ks>0].mean()
    Ks[Ks == 0] = Ks[Ks>0].mean()
    Kp = np.copy(Ks) / 100.0
    # kss=np.copy(Ks)
    Ks[Z==2]=Ks.min()

    #----------Plot 8

    fig=pl.figure(dpi=100)
    cu.Plot_basin(Ks,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Cond. Hidraulica $Ks$ $[mm.s^{-1}]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    #----------------#VELOCIDADES DE LA LADERA

    ksh=((Ks/3600000.0)*cu.CellSlope*(dxp**2.0))/(3*(Hg*0.9/1000.0)**2)

    #----------Plot 9

    fig=pl.figure(dpi=100)
    cu.Plot_basin(ksh,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Vel. Subsup. $[mm.s^{-1}]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    #Manning

    man,p = wmf.read_map_raster(rutamapMan)
    man = cu.Transform_Map2Basin(man,p)

    #----------Plot 10

    fig=pl.figure(dpi=100)
    cu.Plot_basin(man,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Coef. Manning',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    # Vel. Lad. lineal

    v_lad = (1.41/(man*240))*cu.CellSlope**(1.0/2.0)

    #----------Plot 11

    fig=pl.figure(dpi=100)
    cu.Plot_basin(v_lad,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Vel. Lad. Lineal $[mm.s^{-1}]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    # Vel. Lad. no lineal, coeficientes y exponentes seteables.

    CoefLad = (epsilonLad/man)*(cu.CellSlope**(1.0/2.0))
    ExpLad = (2.0/3.0)*e1Lad

    # Vel. canal

    area = cu.CellAcum * (dxp**2)/1e6 #Tamano de celda al cuadrado
    CoefOCG,ExpOCG = wmf.OCG_param(pend = cu.CellSlope, area = area)

    #----------Plot 12

    fig=pl.figure(dpi=100)
    cu.Plot_basin(CoefOCG*cu.CellCauce,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Vel. Cauce $[mm.s^{-1}]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw)

    if CoefOCG_agr is not None:
        #Se agrega el coefOCG, usando mean o median por tramo hidrologico.
        CoefOCG_ = np.zeros(cu.ncells)
        for i in range(1, cu.nhills+1):
            pos = np.where((cu.hills_own == i)&(cu.CellCauce == 1))[0]
            if CoefOCG_agr== 'Mean':
                CoefOCG_[pos] = np.mean(CoefOCG[pos])
            elif CoefOCG_agr== 'Median':
                CoefOCG_[pos] = np.median(CoefOCG[pos])

        #--------------------Plot 12'
        fig=pl.figure(dpi=100)
        cu.Plot_basin(CoefOCG_*cu.CellCauce,
                      lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Vel. Cauce '+CoefOCG_agr+'_tramo $[mm.s^{-1}]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw)
        # seteo
        CoefOCG=CoefOCG_
    #--------------------#DESLIZAMIENTOS    

    #Lectura y transformacion a formato cuenca
    Map, p = wmf.read_map_raster(rutamapAnguloFric)
    Map = cu.Transform_Map2Basin(Map, p )
    #Quita no data y valores nulos
    Map[Map == noDataP] = np.nan
    Map[np.isnan(Map)] = np.nanmean(Map)
    Map[Map == 0] = np.nanmean(Map)
    #lo mete en el modelo 
    cu.set_Slides(Map, 'FrictionAngle')
    #Plot Para ver como queda 
    fig=pl.figure(dpi=100)
    cu.Plot_basin(Map,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Angulo de friccion $[]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    #Lectura y transformacion a formato cuenca
    Map, p = wmf.read_map_raster(rutamapCohesion)
    Map = cu.Transform_Map2Basin(Map, p )
    #lo mete en el modelo 
    Map[Map == noDataP] = 4
    cu.set_Slides(Map, 'Cohesion')
    #Plot Para ver como queda 
    fig=pl.figure(dpi=100)
    cu.Plot_basin(Map,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Cohesion $[]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    #Lectura y transformacion a formato cuenca
    Map, p = wmf.read_map_raster(rutamapPesoEsp)
    Map = cu.Transform_Map2Basin(Map, p )
    #Quita no data y valores nulos
    Map[Map == noDataP] = np.nan
    Map[np.isnan(Map)] = np.nanmean(Map)
    Map[Map == 0] = np.nanmean(Map)
    #lo mete en el modelo 
    cu.set_Slides(Map, 'GammaSoil')
    #Plot Para ver como queda 
    fig=pl.figure(dpi=100)
    cu.Plot_basin(Map,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Peso Especifico $[Kg.m^{-3}]$',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    slope = wmf.cu.basin_arc_slope(cu.structure, cu.DEM, cu.ncells, wmf.cu.ncols, wmf.cu.nrows)
    #Valores contantes debido a que no hay mapas 
    cu.set_Slides(slope,'Slope')
    fig=pl.figure(dpi=100)
    cu.Plot_basin(slope,
                  lines_spaces=0.025,per_color='k',per_lw=1,colorbarLabel='Arco Pendiente [?]',fig=fig,
                    xy=xy, xy_edgecolor=xy_edgecolor,xy_s=xy_s, xy_lw=xy_lw,
                    rutaShp=rutaShp,
                    ShpIsPolygon=ShpIsPolygon,
                    shpColor=shpColor,shpWidth=shpWidth,
                    shpAlpha=shpAlpha)

    cu.set_Slides(0.5,'FS')

    #----------------# SET DE PARAMETROS EN EL NC Y SPEEDTYPE

    # Parametros que no cambian

    #Almacenamiento capilar y gravitacional 
    cu.set_PhysicVariables('capilar',Hu,0)
    cu.set_PhysicVariables('gravit',Hg,1)
    #como hay tanque gravitacional, se establece que si hay retorno 
    wmf.models.retorno = 1

    # Evaporacion, infiltracion, percolacion y perdidas - verticales.
    cu.set_PhysicVariables('v_coef',Evp,0)
    #Infiltracion, se hace impermeable la ciudad, se pasa a [mm/seg]
    cu.set_PhysicVariables('v_coef',Ks/3600.0,1)
    #Percolacion se pasa a [mm/seg]
    cu.set_PhysicVariables('v_coef',Kp/3600.0,2)
    #Se asume perdidas del sistema iguales a cero 
    cu.set_PhysicVariables('v_coef',0,3)
    
    #Coeficiente y exponente del cauce --- siempre es no lineal.
    cu.set_PhysicVariables('h_coef',CoefOCG,3)
    cu.set_PhysicVariables('h_exp',ExpOCG,3)

    # Guardar  versiones SpeedType

    # Lineal

    if (speedtype == np.array([1,1,1])).all() :
        #Coloca todas las velocidades en lineal 
        cu.set_Speed_type()
        # Coeficientes de velocidad horizontal 
        cu.set_PhysicVariables('h_coef',v_lad,0)
        cu.set_PhysicVariables('h_coef',Ks/3600.0,1)
        cu.set_PhysicVariables('h_coef',Kp/3600.0,2)
        #save nc
        cu.Save_SimuBasin(rutaNC +'_v01.nc', 
                          ruta_dem = rutaDEM,
                          ruta_dir = rutaDIR, SimSlides=SimSlides)

    # No lineal en superficial

    elif (speedtype == np.array([2,1,1])).all():
        #Setea config de versiones de velocidad. 1- lineal, 2- No lineal
        cu.set_Speed_type(speedtype)
        # Coeficientes yexponenete de la ladera
        cu.set_PhysicVariables('h_coef',CoefLad,0)
        cu.set_PhysicVariables('h_exp',ExpLad,0)
        #save nc
        cu.Save_SimuBasin(rutaNC +'_v02.nc', 
                          ruta_dem = rutaDEM,
                          ruta_dir = rutaDIR, SimSlides=SimSlides)

    # No lineal en subsup

    elif (speedtype == np.array([1,2,1])).all():
        #Setea config de versiones de velocidad. 1- lineal, 2- No lineal
        cu.set_Speed_type(speedtype)
        # Coeficientes de velocidad horizontal 
        cu.set_PhysicVariables('h_coef',v_lad,0)
        cu.set_PhysicVariables('h_exp',1,0) # Linealiza la ladera de nuevo 
        #El flujo sub-superficial se hace no lineal 
        cu.set_PhysicVariables('h_coef',ksh,1)
        cu.set_PhysicVariables('h_exp',2.0,1)
        #save nc
        cu.Save_SimuBasin(rutaNC +'_v03.nc', 
                          ruta_dem = rutaDEM,
                          ruta_dir = rutaDIR, SimSlides=SimSlides)

    # No lineal en todo

    elif (speedtype == np.array([2,2,2])).all():
        #Setea config de versiones de velocidad. 1- lineal, 2- No lineal
        cuAMVA.set_Speed_type(speedtype)
        # Coeficientes de velocidad horizontal 
        cuAMVA.set_PhysicVariables('h_coef',CoefLad,0)
        cuAMVA.set_PhysicVariables('h_exp',ExpLad,0)
        #El flujo sub-superficial se hace no lineal 
        cuAMVA.set_PhysicVariables('h_coef',ksh,1)
        cuAMVA.set_PhysicVariables('h_exp',2.0,1)
        #save nc
#         cu.Save_SimuBasin(rutaNC +'_v04.nc', 
#                           ruta_dem = rutaDEM,
#                           ruta_dir = rutaDIR, SimSlides=SimSlides)

    return cu,cu.GeoParameters



#################################################################################3 toolbox

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
    
def round_time2(date,datepastinformat,tw):
    '''
    --------
    Notes:

    Last space of datinformat might be in same time units than tw.
    Only works for rounding resolutions between different units (s-h), not between different amounts of same units (1h -3h)

    '''

    date = pd.to_datetime(date)
    datepastinformat = pd.to_datetime(datepastinformat)
    nextdate = datepastinformat + pd.Timedelta(tw)

    dif2next = nextdate - date
    dif2past = date - datepastinformat
    if dif2next < dif2past:
        return nextdate
    else:
        return datepastinformat
    
#find max reloaded nico
# def FindPeaks(Q, Qmin = np.percentile(Q.values[np.isfinite(Q.values)], 90),tw = pd.Timedelta('12h')):
#     '''Find the peack values of the hydrographs of a serie
#     Params:
#         - Q: Pandas serie with the records.
#         - Qmin: The minimum value of Q to be considered a peak.
#         - tw: size of the ime window used to eliminate surrounding maximum values'''
#     #Find the maximum
#     Qmax = Q[Q>Qmin]
#     QmaxCopy = Qmax.copy()
#     #Search the maxium maximorums
#     Flag = True
#     PosMax = []
#     while Flag:
#         MaxIdx = Qmax.idxmax()
#         PosMax.append(MaxIdx)
#         Qmax[MaxIdx-tw:MaxIdx+tw] = -9
#         if Qmax.max() < Qmin: Flag = False
#     #Return the result
#     return QmaxCopy[PosMax]

#find max reloaded sora
def FindMax(S,umbral,window):
    tw=pd.Timedelta(window)
    #Quienes superan el umbral.
    pos=np.where(S>umbral)[0]
    Evs2 =  pd.Series(S[pos], index= S.index[pos])
    #se escogen los maximos en la ventana
    pos1_5 = pd.to_datetime(np.unique([S[i-tw:i+tw].argmax() for i in Evs2.index[:]]))
    Evsin1_5 = pd.Series(S[pos1_5], index= S[pos1_5].index )
    #se filtra lo que se repita dentro de la ventana.
    deltaT=Evsin1_5.index[1:] - Evsin1_5.index[:-1]
    evsin3=np.where(deltaT >= tw)[0] +1 # si los puntos no estancerquita, siempre se come el primer evento
    #si los puntos no estan cerquita, se agrega. 
    if deltaT[0] >= tw: 
        evsin3 = np.sort(np.append(evsin3,0))

    return Evsin1_5[evsin3]

def df_to_tablepng(df,rutafig=None,figsize=(1,2),dpi=100,fontsize=11,
                   header_color='#40466e', row_colors=['#f1f1f2', 'w'],
                   edge_color='w'):

    fig, ax = plt.subplots(figsize=(1,2),dpi=dpi)
    ax.axis('off')
    table = ax.table(cellText=np.round(df.values,2),bbox=[0, 0, 3, 3],#colLabels=df_morfo.columns
             rowLabels=df.index)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # color y formato de letras por fila.
    for k, cell in  six.iteritems(table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0: #or k[1] < 1:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    
    #guardar imagen.
    if rutafig is not None:
        pl.savefig(rutafig,bbox_inches='tight',dpi=dpi,facecolor='w')


####################################################################### WMF toolbox
def getWMFpos_from_coords(cu,coords):
    '''
    cu = SimuBasin instance
    coords = Dataframe, shape = (no. points,2)
    ------
    Return: 
    0. df: index= IDpoints, values = IDsimubasincell
    1. array: longitudes not found.
    2. array: latitudes not found.
    '''
    Cols = [];Fils = []
    #se calcula las coord dentro del simubasin, de cada estacion pluvio
    for i in np.array(coords): 
        col, fil = wmf.cu.coord2fil_col(i[0], i[1])
        Cols.append(col);Fils.append(fil)
    Cols = np.array(Cols);Fils = np.array(Fils)

    #se buscan las coord de cada pluvio como pos del simubasin
    pos_xy=[]; x_no=[];y_no=[]
    for i,j in zip(Cols, Fils):
        pos = np.where((cu.structure[1] == i) & (cu.structure[2] == j))[0]
        if len(pos)!=0:
            #estos son las pos de  las puntos escogidos!!!!!!
            pos_xy.append(int(pos))
        else:
            #puntos no encontrados
            x_no.append(i)
            y_no.append(j)
    
    return pd.DataFrame(np.array([pos_xy]).T,index=coords.index,columns=['wmf_pos']),x_no,y_no
        
############################# graficar var_simubasin por laderas
# cu = cu
# umbral= 'lol'#
# ##############################333
# cu.GetGeo_Cell_Basics()
# cu.GetGeo_Parameters()
# cu.set_Geomorphology(stream_width=cu.CellLong, )
# #Definicion de puntos de control = nodos
# cauce,nodos,n_nodos = wmf.cu.basin_subbasin_nod(
#     cu.structure,
#     cu.CellAcum,
#     umbral,
#     cu.ncells)
# #Las posiciones donde los puntos de control no son la salida de la cuenca
# # esta si es la posicion de salida de cada tramo_ladera
# pos = np.where(nodos!=0)[0]
# #orden de las posiciones
# order_pos=[]
# for pos_ in pos:
#     order_pos.append(cu.hills_own[pos_])

# def vecHills2vecSimubasin(cu,var_x_ladera,x_tramo=False):
#     var_simubasin=np.zeros(cu.ncells)
#     var_simubasin[cu.CellCauce==1]=1
#     for w,poslad in zip(var_x_ladera,order_pos):
#         if x_tramo:
#             var_simubasin[(var_simubasin==1)&(cu.hills_own==poslad)]=w
#         else:
#             var_simubasin[cu.hills_own==poslad]=w
#     return var_simubasin