# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:42:11 2024

@author: Judith
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
import simplekml
import scipy.stats
from sklearn.metrics import r2_score
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.dates as mdates

# from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.gridspec import GridSpec
import tilemapbase


dict_color_instr = {'G2301': '#fb7b50',
              'G4302': '#00698b',
              'Aeris': '#91e1d7',
              'MIRO': '#ffc04c',
              'Aerodyne': '#00b571',
              'LGR': 'firebrick',
              'G2401': 'deeppink',
              'Licor': 'rebeccapurple'}

dict_color_city = {'Rotterdam': 'orange',
              'Utrecht': 'orchid',
              'TorontoDay1-bike': 'deepskyblue',
              'TorontoDay1-car': '#19e8dc',
              'TorontoDay2-car': '#2d75b6',
              'LondonDay2': 'mediumseagreen',
              'LondonDay3': 'darkgreen',
              'LondonDay5': 'lime'}

dict_instr_names = {'Miro': 'MGA10', 
               'Aerodyne': 'Tildas', 
               'G4302': 'G4302',
               'G2301': 'G2301', 
               'G2401': 'G2401',
               'Aeris':'Mira Pico',
               'LGR': 'LGR',
               'Licor': 'LI-7810'}

dict_spec_instr = {'G2301': 'G23',
              'G4302': 'G43',
              'Aeris': 'aeris',
              'Miro': 'miro',
              'Aerodyne': 'aero',
              'LGR': 'LGR',
              'G2401': 'G24',
              'Licor': 'Licor'}





def overview_plot(CH4data,lati,scp_peaks,spec,N,bg=None,th=None,savepath=None):
        fig, ax = plt.subplots(figsize=(18,10))
        plt.plot(CH4data,alpha=.6, label='data')
        
        if bg is not None:
            plt.plot(bg, label = 'background')
        
        plt.plot(th, label='10% threshold')
            
        #CH4data.iloc[scp_peaks].plot(
            #style = 'v', lw=10, color=(1, 0, 0, 1), label = 'all peaks'); #gives error: ValueError: RGBA sequence should have length 3 or 4
        # Plot peaks separately
        plt.plot(CH4data.iloc[scp_peaks].index, CH4data.iloc[scp_peaks].values, 'v', lw=10, label='all peaks')
    
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
        plt.legend()
        plt.xlabel('UTC')
        plt.title(f'{spec}: {str(N)} peaks')
        plt.plot(bg + 0.5, label='0.6 ppm elevation')
        plt.ylim([0, 5])
        if not savepath == None:
            plt.savefig(savepath)
            pass
        plt.show()
        return
    
def overview_plot_ver2(CH4data, lati, scp_peaks, spec, N, bg=None, th=None, savepath=None):
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # Plot CH4 data sul primo asse y
    ax1.plot(CH4data, alpha=0.6, label='CH4 data')
    if bg is not None:
        ax1.plot(bg, label='Background')
    if th is not None:
        ax1.plot(th, label='10% Threshold')
    
    # Plot dei picchi
    ax1.plot(CH4data.iloc[scp_peaks].index, CH4data.iloc[scp_peaks].values, 'v', lw=10, label='All Peaks')

    # Secondo asse y per la latitudine
    ax2 = ax1.twinx()
    ax2.plot(lati, color='orange', alpha=0.6, label='GPS Latitude')
    ax2.set_ylabel('Latitude')

    # Formattazione asse x
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    
    # Label e titolo
    ax1.set_xlabel('GMT')
    ax1.set_ylabel('CH4 Concentration')
    ax1.set_ylim([0, 5])
    ax1.set_title(f'{spec}: {str(N)} peaks')

    # Legenda
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Salvataggio del plot
    if savepath is not None:
        plt.savefig(savepath)
    
    plt.show()

    return
    


def plot_indivpeaks_bevorQC(df_peak_gps, data_gps, path_fig, coord_extent, release_loc1, release_loc2=None, indiv_peak_plots=True, column_names=None, *args):
    
    
    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    t = tilemapbase.tiles.build_OSM()    
    extent = tilemapbase.Extent.from_lonlat(coord_extent[0], coord_extent[1], coord_extent[2], coord_extent[3])
    extent = extent.to_aspect(1.0)
    plotter = tilemapbase.Plotter(extent, t, width=600)
    peak_loc_all = []
    
    
    #data_gps.index = data_gps.index.round('1s')
    df_peak_gps.index = pd.to_datetime(df_peak_gps.index)
  
    
    for i, (index, row) in enumerate(df_peak_gps.iterrows(), start=1):
        starttime = index - timedelta(seconds=12)
        endtime = index + timedelta(seconds=12)
        time = pd.to_datetime(index) #.round('1s')

        if time in data_gps.index:
            lon = data_gps.loc[time]['Longitude']
            lat = data_gps.loc[time]['Latitude']
            coords = (lon, lat)
            peak_loc = coords
            peak_loc_all.append(peak_loc)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            for df, col_name in zip(args, column_names.values()): #, color in ,colors.values()
                ax1.plot(df[col_name][starttime:endtime],color = dict_color_instr[df.name],linewidth=2, label=df.name) #,color=color

            ax1.set_ylabel(r'CH$_4$ Elevation above BG [ppm]')
            ax1.set_xlabel(r'Time')
            ax11    = ax1.twinx()    
            
            v_max = data_gps['Speed [m/s]'].quantile(0.95)
            lns11 = ax11.plot(data_gps.loc[starttime:endtime,'Speed [m/s]'],linestyle=(0,(1,10)),alpha=0.8,)
            # ax11.set_ylim(0,v_max) # Judith's lim
            ax11.set_ylim(-0.10 * v_max, 1.10 * v_max) 
            ax11.set_ylabel('Speed [m/s]')

            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            plotter.plot(ax2, t)
            x, y = tilemapbase.project(*peak_loc)
            x1, y1 = tilemapbase.project(*release_loc1)
            ax2.scatter(x, y, marker="x", color='red', s=30)
            ax2.scatter(x1, y1, marker="x", color='blue', s=30)
            if release_loc2:
                x2, y2 = tilemapbase.project(*release_loc2)
                ax2.scatter(x2, y2, marker="x", color='blue', s=30)
            
            

            w = row['Width (s)']
            bg = round(row['BG'], 2)
            m = round(row['peak max'] - bg, 2)

            fig.suptitle(f'Peak {i},\n max elevation = {m} ppm, bg = {bg} ppm', fontsize=14)
            ax1.legend()

            plt.savefig(path_fig + f"/Peak_{i}_plot.jpg")
            plt.close()

        else:
            print(f'Timestamp of peak {i} is not included')
            fig, ax1 = plt.subplots(figsize=(12, 6))
            for df, col_name in zip(args, column_names.values()):
                ax1.plot(df[col_name][starttime:endtime], label=df.name)
                
            ax1.set_ylabel('elevation above BG [ppm]')

            w = row['Width (s)']
            bg = round(row['BG'], 2)
            m = round(row['peak max'] - bg, 2)

            fig.suptitle(f'Peak {i},\n max elevation = {m} ppm, bg = {bg} ppm', fontsize=14)
            ax1.legend()

            plt.savefig(path_fig + f"/Peak_{i}_plotQC.jpg")
            plt.close()
            
def wind_components(speed, direction):
    
    direction_rad = np.deg2rad(direction)
    u = speed*np.sin(direction_rad)   
    v = speed*np.cos(direction_rad)
    return u, v

def plot_indivpeaks_wind_bevorQC(df_peak_gps, data_gps, path_fig, coord_extent, release_loc1, release_loc2=None, indiv_peak_plots=True, column_names=None, *args):
    
    
    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    t = tilemapbase.tiles.build_OSM()    
    extent = tilemapbase.Extent.from_lonlat(coord_extent[0], coord_extent[1], coord_extent[2], coord_extent[3])
    extent = extent.to_aspect(1.0)
    plotter = tilemapbase.Plotter(extent, t, width=600)
    peak_loc_all = []
    
    
    #data_gps.index = data_gps.index.round('1s')
    df_peak_gps.index = pd.to_datetime(df_peak_gps.index)
  
    
    for i, (index, row) in enumerate(df_peak_gps.iterrows(), start=1):
        starttime = index - timedelta(seconds=12)
        endtime = index + timedelta(seconds=12)
        time = pd.to_datetime(index) #.round('1s')

        if time in data_gps.index:
            lon = data_gps.loc[time]['Longitude']
            lat = data_gps.loc[time]['Latitude']
            coords = (lon, lat)
            peak_loc = coords
            peak_loc_all.append(peak_loc)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            for df, col_name in zip(args, column_names.values()): #, color in ,colors.values()
                ax1.plot(df[col_name][starttime:endtime],color = dict_color_instr[df.name],linewidth=2, label=df.name) #,color=color

            ax1.set_ylabel(r'CH$_4$ Elevation above BG [ppm]')
            ax1.set_xlabel(r'Time')
            ax11    = ax1.twinx()    
            
            v_max = data_gps['Speed [m/s]'].quantile(0.95)
            lns11 = ax11.plot(data_gps.loc[starttime:endtime,'Speed [m/s]'],linestyle=(0,(1,10)),alpha=0.8,)
            ax11.set_ylim(0,v_max)
            ax11.set_ylabel('Speed [m/s]')

            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            plotter.plot(ax2, t)
            x, y = tilemapbase.project(*peak_loc)
            x1, y1 = tilemapbase.project(*release_loc1)
            ax2.scatter(x, y, marker="x", color='red', s=30)
            ax2.scatter(x1, y1, marker="x", color='blue', s=30)
            '''
            print(row['WIND SPEED - LOC 1'])
            if not pd.isna(row['WIND SPEED - LOC 1']):
                print('Sono in IF\n')
                u1, v1 = wind_components(row['WIND SPEED - LOC 1'], row['WIND DIR - LOC 1'])
                ax2.quiver(x1, y1, u1, v1)
'''
            if release_loc2:
                x2, y2 = tilemapbase.project(*release_loc2)
                ax2.scatter(x2, y2, marker="x", color='blue', s=30)
            '''              
                # Add wind speed and direction arrow for loc2
                if not pd.isna(row['WIND SPEED - LOC 2']):
                    u2, v2 = wind_components(row['WIND SPEED - LOC 2'], row['WIND DIR - LOC 2'])
                    ax2.quiver(x2, y2, u2, v2)
'''            

            w = row['Width (s)']
            bg = round(row['BG'], 2)
            m = round(row['peak max'] - bg, 2)

            fig.suptitle(f'Peak {i},\n max elevation = {m} ppm, bg = {bg} ppm', fontsize=14)
            ax1.legend()

            plt.savefig(path_fig + f"/Peak_{i}_plot.jpg")
            plt.close()
            plt.show()

        else:
            print(f'Timestamp of peak {i} is not included')
            fig, ax1 = plt.subplots(figsize=(12, 6))
            for df, col_name in zip(args, column_names.values()):
                ax1.plot(df[col_name][starttime:endtime], label=df.name)
                
            ax1.set_ylabel('elevation above BG [ppm]')

            w = row['Width (s)']
            bg = round(row['BG'], 2)
            m = round(row['peak max'] - bg, 2)

            fig.suptitle(f'Peak {i},\n max elevation = {m} ppm, bg = {bg} ppm', fontsize=14)
            ax1.legend()

            plt.savefig(path_fig + f"/Peak_{i}_plotQC.jpg")
            plt.close()
            
            
def plot_indivpeaks_bevorQC_new(df_peak_gps, data_gps, path_fig, coord_extent, release_loc1, release_loc2=None, indiv_peak_plots=True, column_names=None, *args):
    
    
    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    t = tilemapbase.tiles.build_OSM()    
    extent = tilemapbase.Extent.from_lonlat(coord_extent[0], coord_extent[1], coord_extent[2], coord_extent[3])
    extent = extent.to_aspect(1.0)
    plotter = tilemapbase.Plotter(extent, t, width=600)
    peak_loc_all = []
    
    
    #data_gps.index = data_gps.index.round('1s')
    df_peak_gps.index = pd.to_datetime(df_peak_gps.index)
  
    
    for i, (index, row) in enumerate(df_peak_gps.iterrows(), start=1):
        starttime = row['Peakstart'] - timedelta(seconds=10) #4
        endtime = row['Peakend'] + timedelta(seconds=10)
        time = pd.to_datetime(index) #.round('1s')
        #speed_mean = data_gps.loc[time-timedelta(seconds=4) : time+timedelta(seconds=4),'Speed [m/s]'].mean()
        
        if time in data_gps.index:
            lon = data_gps.loc[time]['Longitude']
            lat = data_gps.loc[time]['Latitude']
            coords = (lon, lat)
            peak_loc = coords
            peak_loc_all.append(peak_loc)
            

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            for df, col_name in zip(args, column_names.values()): #, color in ,colors.values()
                ax1.plot(df[col_name][starttime:endtime],color = dict_color_instr[df.name],linewidth=2, label=df.name) #,color=color

            ax1.axvline(row['Peakstart'], linestyle='-.', color='grey',label='Peakstart&end')
            ax1.axvline(time, linestyle=':', color='grey',label='Peak max')
            ax1.axvline(row['Peakend'], linestyle='-.', color='grey')
            ax1.set_ylabel(r'CH$_4$ Elevation above BG [ppm]')
            ax1.set_xlabel(r'Time')
            ax11    = ax1.twinx()    
            
            v_max = data_gps['Speed [m/s]'].quantile(0.95)
            lns11 = ax11.plot(data_gps.loc[starttime:endtime,'Speed [m/s]'],linestyle=(0,(1,10)),alpha=0.8,)
            ax11.set_ylim(0,v_max)
            ax11.set_ylabel('Speed [m/s]')

            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            plotter.plot(ax2, t)
            x, y = tilemapbase.project(*peak_loc)
            x1, y1 = tilemapbase.project(*release_loc1)
            ax2.scatter(x, y, marker="x", color='red', s=30)
            ax2.scatter(x1, y1, marker="x", color='blue', s=30)
            if release_loc2:
                x2, y2 = tilemapbase.project(*release_loc2)
                ax2.scatter(x2, y2, marker="x", color='blue', s=30)
            
            

            w = row['Width (s)']
            bg = round(row['BG'], 2)
            m = round(row['peak max'] - bg, 2)

            fig.suptitle(f'Peak {i},\n max elevation = {m} ppm, bg = {bg} ppm', fontsize=14)
            ax1.legend()

            plt.savefig(path_fig + f"/Peak_{i}_plot.jpg")
            if (i<240):
                plt.close()
            
            # # only for QC purposes
            # if (i>240):
            #     break

        else:
            print(f'Timestamp of peak {i} is not included')
            fig, ax1 = plt.subplots(figsize=(12, 6))
            for df, col_name in zip(args, column_names.values()):
                ax1.plot(df[col_name][starttime:endtime], label=df.name)
                
            ax1.set_ylabel('elevation above BG [ppm]')

            w = row['Width (s)']
            bg = round(row['BG'], 2)
            m = round(row['peak max'] - bg, 2)

            fig.suptitle(f'Peak {i},\n max elevation = {m} ppm, bg = {bg} ppm', fontsize=14)
            ax1.legend()

            plt.savefig(path_fig + f"/Peak_{i}_plotQC.jpg")
            plt.close()
            
            
            
          
def plot_indivpeaks_afterQC(corrected_peaks, path_fig, coord_extent, release_loc1, release_loc2=None, indiv_peak_plots=True, column_names=None, *args):

    # Define a dictionary mapping city names to releaserate functions
    from peak_analysis.find_analyze_peaks import releaserate_R,releaserate_U,releaserate_L,releaserate_T
    releaserate_functions = {
        'Rotterdam': releaserate_R,
        'Utrecht': releaserate_U,
        'Toronto': releaserate_T,
        'London': releaserate_L
    }
    
    passed_peaks = corrected_peaks.loc[(corrected_peaks['QC'] == True)]
    
    df_main = args[0]['df']
    speed_max = df_main['Speed [m/s]'].quantile(0.97)
    
    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    t = tilemapbase.tiles.build_OSM()    
    extent = tilemapbase.Extent.from_lonlat(coord_extent[0], coord_extent[1], coord_extent[2], coord_extent[3])
    extent = extent.to_aspect(1.0)
    plotter = tilemapbase.Plotter(extent, t, width=600)
    
    
    num_instr = 0
    list_spec = []
    for vars_instr in args:
        num_instr += 1
        list_spec.append(vars_instr['title'])
        city = vars_instr['city']
        day = vars_instr['day']
        print(city)
    print('Outside loop:')
    print(city)
    print('------')
    # Call the releaserate function based on the city name
    releaserate_function = releaserate_functions.get(city) # depending on city (and measurment day, choose different releaserate_functions)
    
    
    for index1, row in passed_peaks.iterrows():
        
        
        fig = plt.figure(figsize=(12,6))
        
        gs = GridSpec(1, 2, width_ratios=[8, 4], figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # First subplot on the left
        ax2 = fig.add_subplot(gs[0, 1])  # Second subplot on the right
        ax11 = ax1.twinx()  # Twin axis for ax1
        
        # Adjust the white margins using subplots_adjust
        plt.subplots_adjust(left=0.07, right=0.97,top=0.85,bottom=0.1)   
        
        trans_s         = row['Peakstart_QC']
        trans_e         = row['Peakend_QC']
        peakno          = row['Peak']
        loca            = row['Loc']
        dx_calc         = round(row['dx_calc_meanspeed'])
        dx_gps          = round(row['dx_gps'])
        
        areas_mean = []
        maxima = []
        for instr in list_spec:
            areas_mean.append(round(row['Area_mean_'+dict_spec_instr[instr]]))
            maxima.append(round(row['Max_'+dict_spec_instr[instr]],1))
        
        mspeed = round(row['Mean_speed'],1)
    
        data_transects = []
        data_names = []
        for vars_instr in args:
            df_instr = vars_instr['df']
            data_transects.append(df_instr[(df_instr.index >= trans_s)&(df_instr.index<=trans_e)])
            data_names.append(vars_instr['title'])
            
    
        time        = pd.to_datetime(index1).round('1s')
        print(time)
        
        # Get release rate at that time and location
        if city == 'London':
            rr_str, rr_fl,r_height  = releaserate_function(index1, loca, day)
        else:
            rr_str, rr_fl = releaserate_function(index1, loca, day)
                
        lon         = row['Longitude']
        lat         = row['Latitude']
        # lons        = dat_G43['Longitude']
        # lats        = dat_G43['Latitude']
        lons        = data_transects[0]['Longitude']
        lats        = data_transects[0]['Latitude']
        peak_loc    = (lon,lat)
        transect    = (lons,lats)
        print(peak_loc)
    
        lns_list = []
        for i in range(num_instr):
            df = data_transects[i]
            name = data_names[i]
            lns1a = ax1.plot(df.loc[:,'CH4_ele_'+dict_spec_instr[name]],color=dict_color_instr[name],
                     label = dict_instr_names[name])
            lns_list.append(lns1a)
            
    
        lns11 = ax11.plot(data_transects[0].loc[:,'Speed [m/s]'], linestyle=(0,(1,10)), alpha=0.8) #label='Speed'
    
        lns  = sum(lns_list, [])
        #lns  = lns1a+lns1b+lns1c #+lns11
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel(r'$\mathrm{CH_4}$ elevation [ppm]')
        ax11.set_ylabel('Speed [m/s]')
        ax11.set_ylim(0,speed_max)
        
        
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        plotter.plot(ax2, t)
        x,y         = tilemapbase.project(*peak_loc)
        x1,y1       = tilemapbase.project(*release_loc1)
        if release_loc2:
            x2,y2       = tilemapbase.project(*release_loc2)
        trans_dx,trans_dy = [],[]
        for i in range(len(transect[0])):
            dx,dy = tilemapbase.project(*(transect[0][i],transect[1][i]))
            trans_dx.append(dx)
            trans_dy.append(dy)
            if i==0:
                dx_0 = dx
                dy_0 = dy
            
        # x3,y3       = tilemapbase.project(*transect)
        ax2.scatter(x,y, marker = "x", color = 'red' ,s = 30, label='peak max')
        ax2.scatter(x1,y1, marker = "x", color = 'blue' ,s = 30, label='release location')
        if release_loc2:
            ax2.scatter(x2,y2, marker = "x", color = 'blue' ,s = 30)
        
        ax2.plot(trans_dx,trans_dy,'r:')
        ax2.scatter(dx_0,dy_0, marker = "o", color = 'yellow' ,s = 10)
        
        fig.suptitle(f'Peak {peakno}' +f' - Loc: {loca}, ' + r'$R_r$: ' +f'{rr_str}',
                     fontweight='semibold')
        str_plot = 'Device, Max. Elevation, Area:'
        for i in range(num_instr):
            name = data_names[i]
            str_plot += f'\n{dict_instr_names[name]}, {maxima[i]} ppm, {areas_mean[i]} ppm*m'
        ax1.set_title(str_plot,loc='left',fontsize=12)
        
        ax2.text(0, 1.02, f'Mean speed: {mspeed} m/s\n' +
                 f'Measured distance: {dx_gps} m\n' +
                 f'Calculated distance: {dx_calc} m',transform=ax2.transAxes, fontsize=12)
        ax2.legend()
        
        plt.savefig(path_fig + f"/Peak_{peakno}_plotQC.png")
        plt.close()
            
    
    
def plot_CH4timeseries(name_city,t_start,t_end, *args,save=False, save_path=None,fig_name=None,column_names=None):
    
    if not column_names:
        column_names = {'Miro': 'CH4_ele_miro', 'Aerodyne': 'CH4_ele_aero', 'G4302': 'CH4_ele_G43','G2301': 'CH4_ele_G23', 'Aeris':'CH4_ele_aeris'}
        print('default CH4 elevation column names assumed')

    
    fig, ax = plt.subplots(figsize=(10, 6))

    for df in args:
        col_name = column_names[df.name]
        ax.plot(df.loc[t_start:t_end].index, df.loc[t_start:t_end,col_name], alpha=0.8,color = dict_color_instr[df.name],linewidth=2, label=dict_instr_names[df.name])

    ax.set_xlabel('UTC', fontsize=14)
    ax.set_ylabel(r'$\mathrm{CH_4}$ elevation [ppm]', fontsize=14)
    ax.set_title(f'{name_city}:' r' $\mathrm{CH_4}$ timeseries', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path+fig_name,bbox_inches='tight')
    plt.show()



def mean_and_median(all_area,all_max):
    all_area = pd.concat(all_area)
    all_max = pd.concat(all_max)
    
    means_area = all_area.groupby('Release_rate').mean()
    means_max = all_max.groupby('Release_rate').mean()
    median_area = all_area.groupby('Release_rate').median()
    median_max = all_max.groupby('Release_rate').median()

    return means_area, means_max, median_area, median_max

def bootstrap(X, Y, b_size):
    # Create an array of indices ranging from 0 to the size of Y - 1
    Index_arr = np.arange(0, np.size(Y), 1)
    
    # Initialize an array to store bootstrap fitting parameters
    bootstrapfits = np.zeros([5, b_size]) #5 see fit
    
    # Perform bootstrapping 'b_size' times
    for i in range(b_size):
        
        # Initialize i_rand with a value that will trigger the first regeneration
        i_rand = np.array([-1])
        # Randomly sample indices from Index_arr with replacement and ensure that not only equal indices are chosen (otherwise no lin. reg. possible)
        while all(x == i_rand[0] for x in i_rand):
            i_rand = np.random.choice(Index_arr, size=np.size(Index_arr), replace=True, p=None)

        # Use the sampled indices to select corresponding X and Y data points
        x_rand = X[i_rand]
        y_rand = Y[i_rand]

        # Fit a linear regression to the randomly sampled data
        fit = scipy.stats.linregress(x_rand, y_rand)
        #fit contains: slope, intercept, r_value, p_value, std_err
        
        # Store the obtained fitting parameters in the bootstrapfits array
        bootstrapfits[:, i] = fit
        
    # Return the array of bootstrap fitting parameters
    return bootstrapfits

def confidenceinterval_bootstrap(X,Y,b_size,conf_level_in_std=1):
    
    bootstrapfits = bootstrap(X,Y,b_size)
    
    #x_intervall = np.linspace(np.min(X),np.max(X),x_numsteps)
    x_intervall = X
    Y_calc = np.zeros([np.size(x_intervall),b_size])
    
    for i in range(0,b_size):
        Y_calc[:,i] = bootstrapfits[1,i]+bootstrapfits[0,i]*x_intervall

    b_mean = np.mean(Y_calc,axis=1)
    b_std = np.std(Y_calc, axis=1)
    b_minussigma = b_mean-conf_level_in_std*b_std
    b_plussigma = b_mean+conf_level_in_std*b_std
    
    #arr_return = np.array([x_intervall,b_mean,b_minussigma,b_plussigma])
    #df_return = pd.DataFrame(data=arr_return, index=None, columns=('x_intervall','b_mean','b_minussigma','b_plussigma'))
    return(bootstrapfits,x_intervall,b_mean,b_minussigma,b_plussigma)

def add_confidenceinterval_to_plot(X,Y,b_size,ax,legend_handle=None,legend_loc=None,conf_level_in_std=1,plot_mean=False,color_fill='lightcoral'):
    
    bootstrapfits,x_intervall,b_mean,b_minussigma,b_plussigma = confidenceinterval_bootstrap(X,Y,b_size,conf_level_in_std=2)

    df_plot = pd.DataFrame([x_intervall,b_mean,b_minussigma,b_plussigma]).T
    # Set column names
    df_plot.columns = ['Release_rate', 'b_mean', 'b_minussigma', 'b_plussigma']
    df_plot_sorted = df_plot.sort_values(by='Release_rate', ascending=True)

    if plot_mean:
        ax.plot(df_plot_sorted.Release_rate,df_plot_sorted.b_mean,c='darkred') #,linewidth=2
    # ax.scatter(df_plot_sorted.release_rate,df_plot_sorted.b_minussigma,marker='_',c='darkred')
    # ax.scatter(df_plot_sorted.release_rate,df_plot_sorted.b_plussigma,marker='_',c='darkred')
    ax.fill_between(df_plot_sorted.Release_rate,df_plot_sorted.b_plussigma,df_plot_sorted.b_minussigma,alpha=0.4,color=color_fill,label=f'{conf_level_in_std} sigma confidence level')
    
    if legend_handle:
        # Create a proxy artist for the legend
        legend_fill = plt.Rectangle((0, 0), 1, 1, fc=color_fill, alpha=0.4)
        if legend_loc:
            if legend_loc == 'inner':
                legend_fill.set_label(f'{conf_level_in_std}'r'$\sigma$' ' confidence\nlevel')
            else:
                print(f'legend_loc must be either None or inner, but given was {legend_loc}')
        else:
            legend_fill.set_label(f'{conf_level_in_std}'r'$\sigma$ confidence level')
        legend_handle.append(legend_fill)
    

def scalar_multiplication(x, a):
    return a*x

def r_squared(func, popt, x, y_data):
    # Calculate predicted values based on the fitted parameters and x values
    y_pred = func(x, *popt)
    # Calculate residuals
    residuals = y_data - y_pred
    # Calculate the total sum of squares (SST)
    y_mean = np.mean(y_data)
    sst = np.sum((y_data - y_mean) ** 2)
    # Calculate the residual sum of squares (SSE)
    sse = np.sum(residuals ** 2)
    # Calculate R²
    r_squared = np.round((1 - sse / sst),2)
    return r_squared



def plot_lollipop(df,rr,y_lims,daytime=None,path_save=None):
    a = df[df['Release_rate'] == rr]
    a.reset_index(inplace=True,drop=False)
    a['Datetime'] = pd.to_datetime(a['Datetime'])

    # Area ----------------------------------------------------------------------
    fig,ax1 = plt.subplots(figsize=(16,12))
    plt.scatter(a['Datetime'],a['ln(Area)'], color='orange',s=100)
    plt.stem(a['Datetime'],a['ln(Area)'], linefmt='grey', basefmt='black')
    # Or plot relative to mean?

    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('ln(Area)',fontsize=20)
    if daytime:
        plt.title(f'{rr} L/min - {daytime}: timeseries of measured areas',fontsize=22)
    else:
        plt.title(f'{rr} L/min: timeseries of measured areas',fontsize=22)
    plt.ylim(y_lims[0],y_lims[1])
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax1.xaxis.set_major_formatter(date_format)

    if path_save:
        if daytime:
            plt.savefig(path_save+f'R_{rr}Lmin{daytime}_lnArea-vs-time.png',bbox_inches='tight')
            plt.savefig(path_save+f'R_{rr}Lmin{daytime}_lnArea-vs-time.pdf',bbox_inches='tight')
            #plt.savefig(path_save+f'R_{rr}Lmin_lnArea-vs-time.svg',bbox_inches='tight')
        else:
            plt.savefig(path_save+f'R_{rr}Lmin_lnArea-vs-time.png',bbox_inches='tight')
            plt.savefig(path_save+f'R_{rr}Lmin_lnArea-vs-time.pdf',bbox_inches='tight')
        
    # Max ----------------------------------------------------------------------
    fig,ax1 = plt.subplots(figsize=(16,12))
    plt.scatter(a['Datetime'],a['ln(Max)'], color='#5142f5',s=100)
    plt.stem(a['Datetime'],a['ln(Max)'], linefmt='grey', basefmt='black')
    
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('ln(Max)',fontsize=20)
    if daytime:
        plt.title(f'{rr} L/min - {daytime}: timeseries of measured areas',fontsize=22)
    else:
        plt.title(f'{rr} L/min: timeseries of measured areas',fontsize=22)
    plt.ylim(y_lims[2],y_lims[3])

    date_format = mdates.DateFormatter('%H:%M:%S')
    ax1.xaxis.set_major_formatter(date_format)

    if path_save:
        if daytime:
            plt.savefig(path_save+f'R_{rr}Lmin{daytime}_lnMax-vs-time.png',bbox_inches='tight')
            plt.savefig(path_save+f'R_{rr}Lmin{daytime}_lnMax-vs-time.pdf',bbox_inches='tight')
            #plt.savefig(path_save+f'R_{rr}Lmin_lnMax-vs-time.svg',bbox_inches='tight')
        else:
            plt.savefig(path_save+f'R_{rr}Lmin_lnMax-vs-time.png',bbox_inches='tight')
            plt.savefig(path_save+f'R_{rr}Lmin_lnMax-vs-time.pdf',bbox_inches='tight')
    

    
def plot_lollipop_bothinsameplot(df,rr,y_lims,daytime=None,path_save=None):
    a = df[df['Release_rate'] == rr]
    a.reset_index(inplace=True,drop=False)
    a['Datetime'] = pd.to_datetime(a['Datetime'])
    
    # Both in Same Plot -----------------------------------------------------------------------

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,16))
    ax1.scatter(a['Datetime'],a['ln(Area)'], color='orange',s=100)
    ax1.stem(a['Datetime'],a['ln(Area)'], linefmt='grey', basefmt='black')
    ax2.scatter(a['Datetime'],a['ln(Max)'], color='#5142f5',s=100)
    ax2.stem(a['Datetime'],a['ln(Max)'], linefmt='grey', basefmt='black')
    
    # Or plot relative to mean?
    # ax2.scatter(a['Datetime'],(a['ln(Max)']/max(a['ln(Max)'])), color='#5142f5',s=100, label='Max')
    # ax1.scatter(a['Datetime'],(a['ln(Area)']/max(a['ln(Area)'])), color='orange',s=100, label='Area')
    # ax2.stem(a['Datetime'],(a['ln(Max)']/max(a['ln(Max)'])), linefmt='grey', basefmt='black')
    # ax1.stem(a['Datetime'],(a['ln(Area)']/max(a['ln(Area)'])), linefmt='grey', basefmt='black') 

    # Adjust the top margin (space between suptitle and plots)
    plt.subplots_adjust(top=0.92)  # Adjust the value as needed

    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax1.set_xlabel('Time',fontsize=20)
    ax1.set_ylabel('ln(Area)',fontsize=20)
    ax1.set_title('Area',fontsize=20, fontweight='bold')
    ax2.set_xlabel('Time',fontsize=20)
    ax2.set_ylabel('ln(Max)',fontsize=20)
    ax2.set_title('Max',fontsize=20, fontweight='bold')
    plt.suptitle(f'{rr} L/min',fontsize=22,fontweight='bold')
    ax1.set_ylim(y_lims[0],y_lims[1])
    ax2.set_ylim(y_lims[0],y_lims[1])
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)

    if path_save:
        if daytime:
            plt.savefig(path_save+f'R_{rr}Lmin{daytime}_both-vs-time.png',bbox_inches='tight')
            plt.savefig(path_save+f'R_{rr}Lmin{daytime}_both-vs-time.pdf',bbox_inches='tight')
        else:
            plt.savefig(path_save+f'R_{rr}Lmin_both-vs-time.png',bbox_inches='tight')
            plt.savefig(path_save+f'R_{rr}Lmin_both-vs-time.pdf',bbox_inches='tight')
            #plt.savefig(path_save+f'R_{rr}Lmin_both-vs-time.svg',bbox_inches='tight')
            #plt.savefig(path_save+f'R_{rr}Lmin_both-vs-time_withouttitle.pdf',bbox_inches='tight')








