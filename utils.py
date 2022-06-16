import glob
import matplotlib.pyplot as plt
import os.path
import pandas as pd
from matplotlib import cm
from sklearn import linear_model
import numpy as np
from scipy import signal, stats
from collections import Counter
import sys

def read_files(csvlist, regions):
    '''Concatenates a list of CSV files as a dataframe.
       For CORDEX project we select those regions which overlap larger than 80%'''
    #csvdata = [pd.read_csv(csv, comment = '#') for csv in csvlist]
    csvdata = []
    for csv in csvlist:
        file = pd.read_csv(csv, comment = '#')
        if 'CORDEX-' in csv.split('/')[-2]:
            domain = csv.split('/')[-2].split('-')[1].split('_')[0]
            file = file[['date'] + regions[domain]]
        csvdata.append(file)
    csvdata = pd.concat(csvdata)
    csvdata['date'] = pd.to_datetime(csvdata['date'], format='%Y-%m')
    return(csvdata.set_index('date'))

def get_average(df, period, months, region = 'world', season = 'Annual'):
    '''Computes the seasonal average for a given region and period'''
    if not region in df.columns:
        rval = np.nan
        res = rval
    else:
        rval = df.loc[period,region]
        res = rval[rval.index.month.isin(months[season])].mean()
    return res

def get_run(filepath, project):
    '''Extracts member identifier from filename'''
    if project == 'CORDEX':
        res = filepath.split('/')[-1].split('_')[1]
    else:
        res = filepath.split('/')[-1].split('_')[2]
    return res

def get_model(filepath, project):
    '''Extracts model name from filename'''
    if project == 'CORDEX':
        res = filepath.split('/')[-1].split('_')[0]
    else:
        res = filepath.split('/')[-1].split('_')[1]
    return res

def get_regional_model(filepath, project):
    '''Extracts regional model name and version from filename'''
    return filepath.split('/')[-1].split('_')[3] + "_" + filepath.split('/')[-1].split('_')[4].split('.')[0]

def decadal_anomalies(basedir, project, period, scenario, Reg_CORDEX_dom, months, region = 'world', season = 'Annual', relative = False):
    data = pd.DataFrame(columns = ['GCM', 'run', 'decade', region])
    if project == 'CORDEX':
        files_scen = glob.glob(f'{basedir}/*_{scenario}*.csv')
    else:
        files_scen = glob.glob(f'{basedir}/{project}_*_{scenario}*.csv')

    for scenfile in files_scen:
        histfile = scenfile.replace(scenario, 'historical')
        if not os.path.exists(histfile):
            #print(f'Missing historical file for {scenfile}')
            continue
        if project == 'CORDEX':
            CORDEX_domain = histfile.split('/')[-2].split('_')[0].split('-')[1]
        
        member_data = read_files([histfile, scenfile], Reg_CORDEX_dom)
        reference_value = get_average(member_data, period, months, region, season)
        if np.isnan(reference_value):
            #print(f'No reference region {region} for {scenfile}')
            continue
        # get decadal anomalies w.r.t. reference
        for decade_start in range(2010,2099,10):
            decade = slice(str(decade_start), str(decade_start+9))
            if project == 'CORDEX':
                dfrow = dict(
                    GCM = get_model(scenfile, project),
                    run = get_run(scenfile, project),
                    RCM_version = get_regional_model(scenfile, project),
                    decade = f'{decade.start}-{decade.stop}',
                    domain = CORDEX_domain
                    )
            else:
                dfrow = dict(
                    GCM = get_model(scenfile, project),
                    run = get_run(scenfile, project),
                    decade = f'{decade.start}-{decade.stop}'
                    )
            dfrow[region] = get_average(member_data, decade, months, region, season) - reference_value
            if relative:
                dfrow[region] = 100. * dfrow[region] / reference_value
            data = data.append(dfrow, ignore_index=True)
    #return data
    if project == 'CORDEX':
        res = data.set_index(['GCM', 'run', 'decade', 'RCM_version', 'domain'])
    else:
        res = data.set_index(['GCM', 'run', 'decade'])
    return res

def GWL_function(root, variable, mask, project, project_gsat, scenario, region, season, period_PI, period_CP, relative, Reg_CORDEX_dom, months):
    
    data_path = 'datasets-aggregated-regionally/data/'
    
    if project == 'CORDEX':
        basedir = root + data_path + project + '/' +  project + '-*_' + variable + '_' +  mask
    else:
        basedir = root + data_path + project + '/' +  project + '_' + variable + '_' +  mask
    basedir_gsat = root + data_path + project_gsat + '/' +  project_gsat + '_tas_' + mask
    
    ydata = decadal_anomalies(basedir, project, period_CP, scenario, Reg_CORDEX_dom, months, region, season, relative = relative)
    xdata = decadal_anomalies(basedir_gsat, project_gsat, period_PI, scenario, Reg_CORDEX_dom, months)

    if project == 'CORDEX':
        pos_GCM_x = [n for n, name in enumerate(xdata.index.names) if name == 'GCM'][0]
        pos_run_x = [n for n, name in enumerate(xdata.index.names) if name == 'run'][0]
        pos_decade_x = [n for n, name in enumerate(xdata.index.names) if name == 'decade'][0]
        GCM_run_decade_xdata = [n[pos_GCM_x] + '_' + n[pos_run_x] + '_' + n[pos_decade_x] for n in xdata.index]

        pos_GCM_y = [n for n, name in enumerate(ydata.index.names) if name == 'GCM'][0]
        pos_run_y = [n for n, name in enumerate(ydata.index.names) if name == 'run'][0]
        pos_decade_y = [n for n, name in enumerate(ydata.index.names) if name == 'decade'][0]
        pos_RCM_y = [n for n, name in enumerate(ydata.index.names) if name == 'RCM_version'][0]
        GCM_run_decade_ydata = [n[pos_GCM_y] + '_' + n[pos_run_y] + '_' + n[pos_decade_y] for n in ydata.index]

        ## GCM_name for xdata is incompleated the I remplace it by GCM_name for ydata
        GCM_run_decade_xdata_new = []
        for x_com in GCM_run_decade_xdata: 
            x_com_new = np.unique([y_com for y_com in GCM_run_decade_ydata if x_com in y_com])
            if x_com_new:
                GCM_run_decade_xdata_new.append(x_com_new[0])
            else:
                GCM_run_decade_xdata_new.append(x_com)

        model = []; run = []; decade = []; xvalue = []; yvalue = []; rcm_version =  []; domain = []
        for n_m, mod in enumerate(ydata.index):
            nn_C = [n_c for n_c, C_m in enumerate(GCM_run_decade_xdata_new) if mod[pos_GCM_y] + '_' + mod[pos_run_y] + '_' + mod[pos_decade_y] in C_m]
            if nn_C:
                model.append(mod[pos_GCM_y]); 
                run.append(mod[pos_run_y]); 
                decade.append(mod[pos_decade_y]); 
                rcm_version.append(mod[pos_RCM_y])
                yvalue.append(ydata.values[n_m][0]);  
                xvalue.append(xdata.iloc[nn_C].values[0][0])
                domain.append(mod[4])

        Dataframe = pd.DataFrame()
        Dataframe['GCM'] = model
        Dataframe['run'] = run
        Dataframe['decade'] = decade
        Dataframe['RCM_version'] = rcm_version
        Dataframe['domain'] = domain
        Dataframe[xdata.columns[0]] = xvalue
        Dataframe[ydata.columns[0]] = yvalue
        Dataframe = Dataframe.set_index(['GCM', 'run', 'RCM_version', 'decade', 'domain'])
        data = Dataframe.copy()
        data = data[~data.index.duplicated(keep='first')]# drop same simulations for different domain 
        #(by defaul eliminate first occurence) 

    else: 
        data = pd.concat([xdata, ydata], axis=1, join='inner').dropna()

    return data

def picture(data_CMIP5_CMIP5, data_CORDEX_CMIP5, data_CMIP5_CMIP5_relative, data_CORDEX_CMIP5_relative,
            data_CMIP5_CMIP5_no_common,
            region, variable, project_gsat, season, mask, path_ATLAS, 
            Reg_CORDEX_dom, units, longname):
            
    fig, ax = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
    
    if variable == 'tas':
        data = data_CMIP5_CMIP5.dropna(axis=0)
    elif variable == 'pr':
        data = data_CMIP5_CMIP5_relative.dropna(axis=0)

    decades_pos = [n for n, name in enumerate(data.index.names) if name == 'decade'][0]
    decades = sorted(list(set(data.index.get_level_values(decades_pos))))
    colors = cm.get_cmap('viridis_r', len(decades))
    ## Plotting all cmip5
    for nsss, sss in enumerate(data_CMIP5_CMIP5_no_common.index):
        xx = data_CMIP5_CMIP5_no_common['world'].iloc[nsss]
        yy = data_CMIP5_CMIP5_no_common[region].iloc[nsss]
        ax[0].scatter(x=xx, y=yy, color='w', edgecolors='k', alpha=0.5, s=10)
    
    h = []
    for k,decade in enumerate(decades):
        decdata = data.xs(decade, level=decades_pos)
        h.append(ax[0].scatter(x=decdata['world'], y=decdata[region], color=colors(k)))
    # Set legend outside the plot
    # Add decadal means
    decadal_means = data.groupby('decade').mean()
    #rb, col, sty = robustness(data, region)
    col = robustness_IPCC(data_CMIP5_CMIP5.dropna(axis=0), region, 'CMIP5', variable, mask, path_ATLAS, Reg_CORDEX_dom)
    # Add linear fit
    lm = linear_model.LinearRegression()
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['world'].values,data[region])
    X = data['world'].values.reshape(-1, 1)
    y = data[region]
    model = lm.fit(X, y)
    ax[0].set_xlim(left=0)
    if p_value<=0.01: sig = '(**)'
    elif (p_value>0.01) and  (p_value<=0.05): sig = '(*)'
    else: sig = ''
    ax[0].text(0.05, 0.92, '$\\beta:$ %.2f %s/K%s\n$R^2:$ %4.2f' % (model.coef_, units[variable], sig, lm.score(X, y)), color='k',
            verticalalignment='top', horizontalalignment='left', transform=ax[0].transAxes, 
            fontsize=13, fontweight = 'bold', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, zorder=50)
    ax[0].plot(X, model.predict(X), color = 'k', linewidth = 2)
    for n_coll, coll in enumerate(col):
        if coll == 'cyan':
            ax[0].scatter(decadal_means['world'].iloc[n_coll], decadal_means[region].iloc[n_coll], 
                          s = 40, c = 'w', marker = 's', edgecolors='k', linewidths=1, hatch=6*'\\', zorder=10)
        elif coll == 'magenta':
            ax[0].scatter(decadal_means['world'].iloc[n_coll], decadal_means[region].iloc[n_coll], 
                          s = 40, c = 'w', marker = 's', edgecolors='k', linewidths=1, hatch=6*'x', zorder=10)
        if coll == 'red':
            ax[0].scatter(decadal_means['world'].iloc[n_coll], decadal_means[region].iloc[n_coll], 
                          s = 40, c = 'red', marker = 's', edgecolors='red', zorder=10)
    # Grid and labels
    ax[0].set_axisbelow(True)
    ax[0].grid()
    #ax[0].axline((1, 1), slope=1, ls="--", c="k", zorder = 5)
    ax[0].set_xlabel(f'{project_gsat} GWL (K)')
    ax[0].set_ylabel(f'{season} {variable} ({units[variable]}) Region: {region}')

    if variable == 'tas': data = data_CORDEX_CMIP5.dropna(axis=0)
    elif variable == 'pr': data = data_CORDEX_CMIP5_relative.dropna(axis=0)
    
    decades_pos = [n for n, name in enumerate(data.index.names) if name == 'decade'][0]
    decades = sorted(list(set(data.index.get_level_values(decades_pos))))
    colors = cm.get_cmap('viridis_r', len(decades))
    h = []
    for k,decade in enumerate(decades):
        decdata = data.xs(decade, level=decades_pos)
        h.append(ax[1].scatter(x=decdata['world'], y=decdata[region], color=colors(k)))
    # Set legend outside the plot
    ax[1].legend(h, decades, loc='center left', bbox_to_anchor=(1, 0.64), ncol=1)
    # Add decadal means
    decadal_means = data.groupby('decade').mean()
    col = robustness_IPCC(data_CORDEX_CMIP5.dropna(axis=0), region, 'CORDEX', variable, mask, path_ATLAS, Reg_CORDEX_dom)
    # Add linear fit
    lm = linear_model.LinearRegression()
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['world'].values,data[region])
    X = data['world'].values.reshape(-1, 1)
    y = data[region]
    model = lm.fit(X, y)
    ax[1].set_xlim(left=0)
    if p_value<=0.01: sig = '(**)'
    elif (p_value>0.01) and (p_value<=0.05): sig = '(*)'
    else: sig = ''
    ax[1].text(0.05, 0.92, '$\\beta:$ %.2f %s/K%s\n$R^2:$ %4.2f' % (model.coef_, units[variable], sig, lm.score(X, y)), color='k',
            verticalalignment='top', horizontalalignment='left', transform=ax[1].transAxes, 
            fontsize=13, fontweight = 'bold', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, zorder=50)
    ax[1].plot(X, model.predict(X), color = 'k', linewidth = 2)
    for n_coll, coll in enumerate(col):
        if coll == 'cyan':
            ax[1].scatter(decadal_means['world'].iloc[n_coll], decadal_means[region].iloc[n_coll], 
                          s = 40, c = 'w', marker = 's', edgecolors='k', linewidths=1, hatch=6*'\\', zorder=10)
        elif coll == 'magenta':
            ax[1].scatter(decadal_means['world'].iloc[n_coll], decadal_means[region].iloc[n_coll], 
                          s = 40, c = 'w', marker = 's', edgecolors='k', linewidths=1, hatch=6*'x', zorder=10)
        if coll == 'red':
            ax[1].scatter(decadal_means['world'].iloc[n_coll], decadal_means[region].iloc[n_coll], 
                          s = 40, c = 'red', marker = 's', edgecolors='red', zorder=10)
    # Grid and labels
    ax[1].set_axisbelow(True)
    ax[1].grid()
    ax[1].set_xlabel(f'{project_gsat} GWL (K)')
    
    h_sig = []
    h_sig.append(plt.scatter(x=8, y=1, s = 40, c = 'w', marker = 's', edgecolors='k', linewidths=1, hatch=6*'x', zorder=100))
    h_sig.append(plt.scatter(x=8, y=1, s = 40, c = 'w', marker = 's', edgecolors='k', linewidths=1, hatch=6*'\\', zorder=100))
    h_sig.append(plt.scatter(x=8, y=1, s = 40, c = 'red', marker = 's', edgecolors='red', zorder=100))
    # Set legend outside the plot
    ax[0].legend(h_sig, ['Conflicting signals', 'No change or no robust signal', 'Robust Signal'],
                 loc='center left', bbox_to_anchor=(2.05, 0.16), ncol=1)

    ax[0].set_xlim([0, 6])
    ax[1].set_xlim([0, 6])
    
    ax[0].set_title("CMIP5")
    ax[1].set_title("CORDEX")
    
    plt.subplots_adjust(wspace=0.05)

def select_GCM_from_CORDEX(data_CMIP5_CMIP5, data_CORDEX_CMIP5):
    """Select those GCM models available in CORDEX dataset"""
    GCMs_CMIP5 = np.unique([n[0] for n in data_CMIP5_CMIP5.index])
    GCMs_CORDEX = np.unique([n[0] for n in data_CORDEX_CMIP5.index])
    for GCM_CMIP5 in GCMs_CMIP5:
        match = [GCM_CORDEX for GCM_CORDEX in GCMs_CORDEX if GCM_CMIP5 in GCM_CORDEX]
        if not match:
            posi = [nc for nc, gc in enumerate(data_CMIP5_CMIP5.index) if gc[0] == GCM_CMIP5]
            data_CMIP5_CMIP5 = data_CMIP5_CMIP5.drop(data_CMIP5_CMIP5.index[posi])
    return data_CMIP5_CMIP5

def weight_GCM_from_CORDEX(data_CMIP5_CMIP5, data_CORDEX_CMIP5):
    """Replicate GCMs results based on the number of RCM runs for every GCM (weighting)"""
    data_CMIP5_CMIP5_rep = data_CMIP5_CMIP5.copy()
    n_decades = len(data_CORDEX_CMIP5.groupby('decade').mean().index)
    GCMs_CORDEX = [n[0] for n in data_CORDEX_CMIP5.index][::9]
    GCMs_CMIP5 = [n[0] for n in data_CMIP5_CMIP5.index][::9]
    for GCM in GCMs_CMIP5:
        n_repetitions = [GCM for GCM_CORDEX in GCMs_CORDEX if GCM in GCM_CORDEX]
        if len(n_repetitions) > 1:
            for n_t in np.arange(len(n_repetitions)-1):
                data_CMIP5_CMIP5_rep = pd.concat([data_CMIP5_CMIP5_rep, data_CMIP5_CMIP5.xs(n_repetitions[0], level='GCM', drop_level=False)])
    return data_CMIP5_CMIP5_rep

def robustness_IPCC(data, region, project, variable, mask, root, CORDEX_regions):
    """Calculate the robusteness using the advanced approach used in the IPCC
    (Gutiérrez et al., 2021, Cross-ChapterBoxAtlas.1)"""
    data_robust = data.copy()
    data_robust[region + '_robust'] = np.arange(data.shape[0])*0
    
    for n_model, model in enumerate(data.index):
        if project == 'CORDEX':
            file_hist = root + 'datasets-aggregated-regionally/data/CORDEX/CORDEX-'\
                        + model[4] + '_' + variable + '_' + mask + '/' + model[0] + '_' + model[1]\
                        + '_historical_' + model[2] + '.csv'
        elif project == 'CMIP5':
            files = glob.glob(root + 'datasets-aggregated-regionally/data/CMIP5/CMIP5_'\
                                + variable + '_' + mask + '/CMIP5*')
            for ff in files:
                if 'historical' in ff.split('/')[-1]:
                    if (ff.split('/')[-1].split('_')[1] in model[0]) and (ff.split('/')[-1].split('_')[2] == model[1]):
                        file_hist = ff
        
        # We calculate the variability threshold using the historical period               
        data_hist = pd.read_csv(file_hist, comment = '#', index_col = 0, parse_dates = True)
        if 'CORDEX-' in file_hist.split('/')[-2]:
            domain = file_hist.split('/')[-2].split('-')[1].split('_')[0]
            data_hist = data_hist[CORDEX_regions[domain]]
        period_CP = slice('1971', '2005')
        data_hist = data_hist[data_hist.index.year>=int(period_CP.start)]
        data_hist = data_hist[data_hist.index.year<=int(period_CP.stop)]
        data_hist_annual_mean = data_hist.resample('y').mean()
        # detrending
        data_hist_annual_mean_detrend = signal.detrend(data_hist_annual_mean[region].values)
        th_v = np.sqrt(2)*1.645*data_hist_annual_mean_detrend.std()/np.sqrt(20)
        if abs(data[region].iloc[n_model])>th_v:
            data_robust[region + '_robust'].iloc[n_model] = 1
    
    ## We calculate the number of models agree on sign of change and the number of models which show a change greater than variability
    decadal_mean = data_robust.groupby('decade').mean()

    direction_decade = []
    for n_decade, gr in enumerate(data_robust.groupby('decade')):
        mean_decade_region = decadal_mean[region].iloc[n_decade]
        if mean_decade_region>0:
            direction = 100*np.sum(gr[1][region].values>0)/len(gr[1][region].values)
        elif mean_decade_region<0:
            direction = 100*np.sum(gr[1][region].values<0)/len(gr[1][region].values)
        direction_decade.append(direction)
        
    color = []
    for n_d in range(len(data_robust.groupby('decade'))):
        if decadal_mean[region + '_robust'].iloc[n_d]<0.66: # No change
            color.append('cyan')
        else: 
            if direction_decade[n_d]<80: # conflicting signal
                color.append('magenta')
            else: # Robust signal
                color.append('red')
                
    return color

def main(root, variable, mask, project_gsat, scenario, region, season, period_PI, period_CP):

    # Mosaic-ensemble approach used in the IPCC-WGI AR6 (Diez-Sierra et al., 2022, see Figure 1 panel b)
    Reg_CORDEX_dom = {'NAM' : ['NWN', 'NEN', 'WNA', 'CNA', 'ENA'],
                      'CAM' : ['NCA', 'SCA', 'CAR'],
                      'SAM' : ['NWS', 'NSA', 'NES', 'SAM', 'SWS', 'SES', 'SSA'],
                      'ARC' : ['GIC', 'RAR', 'ARO'],
                      'EUR' : ['NEU', 'WCE', 'MED'],
                      'AFR' : ['SAH', 'WAF', 'CAF', 'NEAF', 'SEAF', 'WSAF', 'ESAF', 'MDG'],
                      'ANT' : ['EAN', 'WAN'],
                      'WAS' : ['WCA', 'TIB', 'SAS', 'ARP'],#'ARS', 'BOB', 'EIO'
                      'SEA' : ['SEA'],
                      'EAS' : ['ECA', 'EAS', 'SAS'],
                      'AUS' : ['NAU', 'CAU', 'EAU', 'SAU', 'NZ']}

    longname = dict(pr = 'Precipitation', tas = 'Near surface temperature')
    units = dict(pr='%', tas='K')
    months = dict(DJF=[1,2,12], MAM=[3,4,5], JJA=[6,7,8], SON=[9,10,11], Annual=range(1,13))

    data_CMIP5_CMIP5_relative = GWL_function(root, variable, mask, 'CMIP5', project_gsat, scenario, region, season, period_PI, period_CP, True, Reg_CORDEX_dom, months)
    data_CORDEX_CMIP5_relative = GWL_function(root, variable, mask, 'CORDEX', project_gsat, scenario, region, season, period_PI, period_CP, True, Reg_CORDEX_dom, months)
    data_CMIP5_CMIP5 = GWL_function(root, variable, mask, 'CMIP5', project_gsat, scenario, region, season, period_PI, period_CP, False, Reg_CORDEX_dom, months)
    data_CORDEX_CMIP5 = GWL_function(root, variable, mask, 'CORDEX', project_gsat, scenario, region, season, period_PI, period_CP, False, Reg_CORDEX_dom, months)

    # selecting commun CORDEX-CMIP5 simulations
    data_CMIP5_CMIP5_relative_common = select_GCM_from_CORDEX(data_CMIP5_CMIP5_relative, data_CORDEX_CMIP5_relative)
    data_CMIP5_CMIP5_common = select_GCM_from_CORDEX(data_CMIP5_CMIP5, data_CORDEX_CMIP5)
    
    # Weighting CMIP5 simulations
    data_CMIP5_CMIP5_common_weight = weight_GCM_from_CORDEX(data_CMIP5_CMIP5_common, data_CORDEX_CMIP5)
    data_CMIP5_CMIP5_relative_common_weight = weight_GCM_from_CORDEX(data_CMIP5_CMIP5_relative_common, data_CORDEX_CMIP5_relative)

    if variable == 'tas': rr_aa = data_CMIP5_CMIP5.copy()
    elif variable == 'pr': rr_aa = data_CMIP5_CMIP5_relative.copy()

    picture(data_CMIP5_CMIP5_common_weight, 
            data_CORDEX_CMIP5, 
            data_CMIP5_CMIP5_relative_common_weight,
            data_CORDEX_CMIP5_relative,
            rr_aa,
            region, variable, project_gsat, season, mask, root,
            Reg_CORDEX_dom, units, longname)   