#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker  #
import pandas as pd
import os 
import sys
from scipy import stats
from scipy.stats import norm
from shutil import get_terminal_size
import warnings
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)

infile = "HI_fit_1-597_uv_bottom=14.80.csv"; uv = 14.8
df = pd.read_csv(infile); print(df.describe())

def Q_plot(data,para1,para2,N_limit,ylabel,ylabel2,log,inc,equal,nbins,limits,SE_SD): 
    def binning(ax):
        arr = []
        dfb = pd.DataFrame()
        censor = 0 # IF WORKING WITH LIMITS 
        dfb['X'] = data[para1]; dfb['Y'] = data[para2]; dfb['censor'] = censor
        dfb = dfb[['censor','X','Y']].reset_index(); del dfb['index']
        #print(dfb)
        bins = int((x2-x1)*inc); start = int(x1)
        binned = []
        df_bin = pd.DataFrame()
        if equal == 'S':  # EQUAL SPREAD 
            for i in range (0,bins+1): 
                s = start + float(i)/inc
                e = start + float(i+1)/inc
                tmp = dfb[(dfb['X'] >= s) & (dfb['X'] < e)].reset_index();
                if len(tmp) > 0:
                    xmax = max(tmp['X']); xmin = min(tmp['X'])
                else:
                    xmax = 0; xmin =0
                x_mean = start + float(i+0.5)/inc; dx = 0.5/inc;
                y_mean = np.mean(tmp['Y']); dy = np.std(tmp['Y'],ddof=1) # sample
                n = len(tmp)
                #if limits == "nolim":
                # STILL TO ADD LIMITS (ASURV) STUFF
                values = [n,x_mean,dx,y_mean,dy,xmin,xmax]
                arr.append(values)
                dfi = pd.DataFrame(arr, columns=['n','x_mean','dx','y_mean','dy','xmin','xmax'])
               
        else:  # EQUAL NUMBER/BIN (QUANTILE)
            dbs = int(len(dfb)/nbins)
            print("%d bins over %d entries - %d per bin with remainder %s"
                  %(bins,len(dfb), dbs, len(dfb) - (nbins*dbs)))
            dfb = dfb.sort_values('X');
            dfb = dfb.reset_index(); dfb.index += 1;
            
            for i in range(0,nbins):
                start = i*(dbs)+1; end = (i+1)*(dbs)
                tmp = dfb.loc[start:end:1]; #print(tmp)
                x_mean = np.mean(tmp['X']); dx = np.std(tmp['X'],ddof=1)
                y_mean = np.mean(tmp['Y']); dy = np.std(tmp['Y'],ddof=1)
                xmax = max(tmp['X']); xmin = min(tmp['X'])
                n = len(tmp)
                values = [n,x_mean,dx,y_mean,dy,xmin,xmax]
                arr.append(values)
                dfi = pd.DataFrame(arr, columns=['n','x_mean','dx','y_mean','dy','xmin','xmax'])

        df_bin = df_bin.append(dfi); #print(df_bin)
        
        df_bin = df_bin[df_bin['n'] > 0]
        x = df_bin['x_mean']; dx = df_bin['dx']; y = df_bin['y_mean'];
        left = x -  df_bin['xmin']; right = df_bin['xmax'] -x
    
        if SE_SD == "SD":
            dy = df_bin['dy']
        else:
            dy = df_bin['dy']/(df_bin['n']**0.5)

        if equal == "N":
            ax.errorbar(x, y, xerr=(left,right), yerr=dy, fmt='.', c = 'k', capsize=2,zorder = 2)
            newx1 = x1; newx2 = x2
        else:
            ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='.', c = 'k', capsize=2,zorder = 2)
            newx1 = x1 - dx[0]; newx2 = x2 + dx[0]/inc # SPACE FOR ERROR BARS
        return newx1,newx2   
    ################ FAKE LOG SCALES ################ 
    def fake_log (ax,xax,yax,xstart,xend,ystart,yend):
        def update_ticks(z, pos):
            return "$10^{%1.0f}$" %(z) 

        def minor_ticks(start,end):
            minor = []
            for i in range (int(start),int(end)):
                for j in range(i,i+1):
                    for k in range(1,10):
                        minor.append(j+np.log10(k))
            return minor

        if xax == 'y':
            axs[ax].xaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
            minor = minor_ticks(xstart,xend); axs[ax].set_xticks(minor, minor=True)
        
        if yax == 'y':
            axs[ax].yaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
            minor = minor_ticks(ystart,yend); axs[ax].set_yticks(minor, minor=True)
       
    ###############  NICE p_value LABEL  #################
    def p_nice(p_in,label):
        p_string = "%1.3e" %(p_in); #print(p_string,p_in)
        if p_in > 1e-200:    
            if p_in < 0.01:
                p1 = p_string[:4]; 
                p2,p3, = p_string.split('e'); 
                if p3[1] == "0":
                    p3 = "-"+p3[2:]
                p_text = r"$p(\%s) = %s \times 10^{%s}$" %(label,p1,p3);
            else:
                p_text = r"$p(\%s) = %1.3f$" %(label,p_in);
        else:
            p_text = r"$p(\%s) \sim 0$" %(label);
        return p_text
    ################################################################
    font = 14
    plt.rcParams.update({'font.size': font})
    fig, axs = plt.subplots(2,1,figsize=(6,6), gridspec_kw={'height_ratios': [1,0.5]})
    plt.setp(axs[0].spines.values(), linewidth=2)
    axs[0].tick_params(direction='in', pad = 7,length=6, width=1.5, which='major',right=True)
    axs[0].tick_params(direction='in', pad = 7,length=3, width=1.5, which='minor',right=True)

    ### NEED TO LOSE THE NaNs FOR STATS 
    data=data[(data[para1] == data[para1]) & (data[para2] == data[para2])]
    print(data)
        
    tau, p_value = stats.kendalltau(data[para1], data[para2])   
    ## CONVERT p_VALUE TO Z-value
    p = p_value/2 # TWO SIDED
    Z = norm.ppf(1-p,loc=0,scale=1)
    print('Kendalls tau = %1.3f p-value = %1.3e => Z = %1.2f' %(tau,p_value, Z))
    
    dets = data[data['flag'] == "det"]; 
    nons = data[data['flag'] == "non"]
    
    x = nons[para1]; y = nons[para2]
    axs[0].scatter(x,y,facecolors="white", edgecolors = "r", s=20);

    x = dets[para1]; y = dets[para2]
    axs[0].scatter(x,y,facecolors="k", edgecolors = "k", s=20);

    x1,x2 = axs[0].get_xlim(); y1,y2 = axs[0].get_ylim();
    if log == "log":
        fake_log(0,'n','y',x1,x2+1,y1,y2+1)
    tick_spacing = 1
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axs[0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    axs[0].set_xlim(x1,x2); axs[0].set_ylim(y1,y2)

    axs[0].set_ylabel(ylabel, size =font)
    axs[0].axes.xaxis.set_ticklabels([])#

    ################## BOTTOM PANEL - BINNING ###############################
    plt.setp(axs[1].spines.values(), linewidth=2)
    axs[1].tick_params(direction='in', pad = 7,length=6, width=1.5, which='major',top=True,right=True)
    axs[1].tick_params(direction='in', pad = 7,length=3, width=1.5, which='minor',top=True,right=True)

    x1,x2 = binning(axs[1]) 
   
    axs[1].set_xlabel(r'Ionising photon rate, $\log_{10}Q_{\rm HI}$ [s$^{-1}$]', size =font)
    axs[1].set_ylabel(ylabel2, size =font)

    y1,y2 = axs[1].get_ylim();
    if log == "log":
        fake_log(1,'n','y',x1,x2+1,y1,y2+1)
    
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    axs[1].set_xlim(x1,x2); axs[0].set_xlim(x1,x2);
    
    xpos = x1+(x2-x1)/24;
    y1t,y2t = axs[0].get_ylim(); ypos = y2t-(y2t-y1t)/12; yskip = (y2t-y1t)/12;
    text = p_nice(p_value,'tau'); 
    axs[0].text(xpos,ypos, r"%s, $n = %d$" %(text,len(data)), fontsize = 0.8*font, c = 'k')
    axs[1].set_ylim(y1,y2)
       
    plt.tight_layout()#
    outfile = '%s-%s_uv_bot=%1.2f-bin=%s_%s.png' %(para1,para2,uv,equal,SE_SD)
    plt.savefig(outfile); print('Written to %s' %(outfile))
    plt.show()
      
Q_plot(df,'Q','TOssd',19,r'Turnover frequency, $\nu_{\rm TO}$ [Hz]',r'$\nu_{\rm TO}$ binned','log',0.5,'S',10,'nolim','SE')
#Q_plot(df,'Q','thick',19, r'Spectral index, $\alpha_{\rm thick}$',r'$\alpha_{\rm thick}$ binned','nolog',0.5,'S',10,'nolim','SE')
# equal = S       EQUALLY SPACED 
# equal = N       NUMBER SAME IN EACH BIN, SHOWING RANGE OF BINNING
# equal = NS      NUMBER SAME IN EACH BIN, SHOWING +/-1 SIGMA IN X 


