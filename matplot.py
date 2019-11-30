#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os
import gzip
import matplotlib.pyplot as plt
import numpy as np
import logging
from pprint import pformat
from matplotlib.colors import LogNorm
from decimal import Decimal
from mnvtf.hdf5_readers import SimpleCategorialHDF5Reader as HDF5Reader

def usage():
    sys.exit('''
Usage:
    python mat_plot.py <(probabilities).py> <(truth-labels).py>
Code to print confusion matrix given the actuals vector ant the probability
vector
Example usage:
    python mat_plot.py <(probabilities).py> <(truth-labels).py> \
        optional <(invariant mass W).py>
or:
    python mat_plot.py <(conf_mat).py>
'''
    )



def file2npy(filename, index_dict):
    data_dict = {}
    LOGGER.info('\nLoading {}...'.format(filename))
    string = filename.split('.')
    ftype = string[len(string)-1]
    
    if ftype == 'gz':
        text_file = gzip.open( filename, 'r' )
        for i, l in enumerate( text_file ):
            pass
        entries = i + 1
        LOGGER.info("There are {} entries.".format(entries))


    elif ftype == 'root':
        from ROOT import TFile, TTree
        mc_file = TFile( filename, 'read' )
        mc_tree = mc_file.Get('evt_pred')
        entries = mc_tree.GetEntries()
    else:
        sys.exit('File extension not supported')
    
    for key in index_dict:
        data_dict[key] = np.zeros(entries)
    
    if ftype == 'gz':
        text_file = gzip.open( filename, 'rt' )
        for i, line in enumerate(text_file):
            line = line.strip()
            currentline = line.split(',')
            
            for key in index_dict:
                data_dict[key][i] = currentline[index_dict[key]]
        
        text_file.close()
        LOGGER.info("Text file loaded")
    elif ftype == 'root':
        entries = mc_tree.GetEntries()
    
        for i, event in enumerate(mc_tree):
            if i % 50000 == 0:
                LOGGER.info('Entry {}/{}'.format(i,entries))
                sys.stdout.flush()
        
            W[i]           = mc_tree.mc_w / 1000
            Q2[i]          = mc_tree.mc_Q2 / 1000000
            n_tracks[i]    = mc_tree.n_tracks
            actuals[i]     = mc_tree.true_mult
            predictions[i] = mc_tree.pred_mult
            
        mc_file.Close()
        
        LOGGER.info("ROOT file loaded")
    
    return data_dict #, W, Q2, n_tracks actuals,



def plot_array(array, hist_title, file_title,
               ann = False, logz = False, dec = False):
    '''
    Read the confusion matrix <array> and plot it. Needed input is the
    title drawn in the plot, and the name of the .pdf output file.
    Also change <ann> to True if you want annotations of bin values on
    top of them. <logz> is to draw log scales, and dec is truncate decimals
    '''

    LOGGER.info("Saving {} plot...".format(hist_title))
    fig, ax = plt.subplots()
    if logz:
        norm = LogNorm()
    else:
        norm = None
    if array.max() > 1:
        img1 = plt.imshow(array, cmap='Reds', norm=norm)
    else:
      img1 = plt.imshow(array, cmap='Reds', vmin=0, vmax=0.8)
    colours = img1.cmap(img1.norm(np.unique(array)))

    #We want to show all ticks...
    axis_max = len(array)
    if axis_max < 15:
        ax.set_xticks(np.arange(axis_max))
        ax.set_yticks(np.arange(axis_max))
        
        #...and label them with the respective list entries
        ax.set_xticklabels(np.arange(axis_max))
        ax.set_yticklabels(np.arange(axis_max))

        #Loop over data dimensions and create text annotations.
        threshold = img1.norm(array.max())/2.
        textcolors = ["black", "white"]
    
    if axis_max < 9:
        for j, row in enumerate(array):  #Columns and rows are same length
            for i, column in enumerate(array):
                if dec:
                    x = round(array[i,j], 3)
                else:
                    x = Decimal(array[i,j])
                try:
                    color = textcolors[img1.norm(array[i,j]) > .35]
                except TypeError:          #We have problems with log norm
                    color = "black"
                text = ax.text(j, i, x,
                       ha="center", va="center", color=color)

    fig.colorbar(img1, ax=ax)              #Color bar is the size of the plot
    plt.ylabel("truth (label)")
    plt.xlabel("prediction (target)")
    plt.title(hist_title)
    plt.tight_layout()
    plt.savefig(file_title)
    plt.close()



def column_normalize(array):
    for j, column in enumerate(array):
        norm = array[:,j].sum(0)
        if norm != 0:
            array[:,j] = array[:,j]/norm
     
    return array



def row_normalize(array):
    for j, row in enumerate(array):
        norm = array[j].sum(0)
        if norm != 0:
            array[j] = array[j]/norm
    
    return array



# Recall, precision and F1 score
def prcsn_rcll(array):
    perf_mat = []
    keyes = ['Mult', 'r', 'p', 'F1']
    perf_mat.append(keyes)
    acc = 0
    f1mean = 0
    f1hmean = 0
    f1valid = True
    
    for j, column in enumerate(array):
        rcll_norm  = array[:,j].sum(0)
        if rcll_norm != 0:
            rcll = array[j,j]/rcll_norm
        else:
            rcll = 0
            
        prcsn_norm = array[j].sum(0)
        
        if prcsn_norm != 0:
            prcsn = array[j,j]/prcsn_norm
        else:
            prcsn = 0
        
        if prcsn+rcll != 0:
            F1 = 2*prcsn*rcll/(prcsn+rcll)
        else:
            F1 = 0
        
        values = [j, round(rcll, 4), round(prcsn, 4), round(F1, 4)]
        perf_mat.append(values)
        
        if F1 != 0 and f1valid:
            f1hmean += 1/F1
        else:
            f1hmean = 0
            f1valid = False
            
        f1mean += F1
        acc += array[j,j]
    
    if f1valid:
        f1hmean = len(array)/f1hmean
        
    f1mean /= len(array)
    acc = float(acc) / float(array.sum())
    
    acc = 100*round(acc,3)
    f1mean = 100*round(float(f1mean),3)
    f1hmean = 100*round(f1hmean,3)

    LOGGER.debug('"Recall", "precision" and, "F1 score" per label:')
    LOGGER.debug(pformat(perf_mat))
    LOGGER.debug( "Global accuracy: {}%".format(acc)  )
    LOGGER.debug( "F1 mean: {}%".format(f1mean) )
    LOGGER.debug( "F1 harmonic mean: {}%\n".format(f1hmean) )
    
    

def conf_mat(actuals=None, predictions=None, suffix='', cm=None):
    dec=True
    ann=True
    
    if '0' in suffix:
        prefix = '1.0 < W < 1.4 [GeV] '
    elif '1' in suffix:
        prefix = '1.4 < W < 2 [GeV] '
    elif '2' in suffix:
        prefix = 'W > 2 [GeV] '
    else:
        prefix = ''
    
    if cm is not None:
        f = cm
    elif actuals is not None and predictions is not None:
        mat_size = np.amax(np.maximum(actuals, predictions)) + 1
        conf_mat = np.zeros((mat_size,mat_size))
        for a, p in zip(actuals, predictions):
            conf_mat[a,p] += 1
        
        f = conf_mat
    else:
        sys.exit("There is nothing to plot")

    plot_array(
        f.copy(),
        prefix +"Confusion Matrix",
        "conf_mat"+suffix+".pdf",
        ann=ann
    )
    plot_array(
        f.copy(),
        prefix +"LogConfusion Matrix",
        "log_conf_mat"+ suffix +".pdf",
        ann=ann, logz=True
    )
    g = column_normalize(f.copy())
    plot_array(
        g,
        prefix +"Column Normalized Confusion Matrix",
        "conf_mat_col_norm"+ suffix +".pdf",
        ann=ann,
        dec=dec
    )
    h = row_normalize(f.copy())
    # printing row norm matrix is usefull for bilinear loss
    if suffix is '':
        LOGGER.debug("\nRow normalized comfusion matrix:\n{}".format(h))
    plot_array(
        h,
        prefix +"Row Normalized Confusion Matrix",
        "conf_mat_row_norm"+ suffix +".pdf",
        ann=ann,
        dec=dec
    )
    fmat = 2*np.where( g*h != 0, np.divide(g*h, g+h), 0)
    plot_array(
        fmat,
        prefix +"F1 Confusion Matrix",
        "conf_mat_f1"+ suffix +".pdf",
        ann=ann,
        dec=dec
    )

    prcsn_rcll(f.copy())



def mult_plot(actuals, predictions, n_tracks=None, suffix=''):
    if suffix == '0':
        prefix = '0.9 < W < 1.4 [GeV] '
    elif suffix == '1':
        prefix = '1.4 < W < 2 [GeV] '
    elif suffix == '2':
        prefix = 'W > 2 [GeV] '
    else:
        prefix = suffix
    LOGGER.info("Saving multiplicities {}plot...".format(prefix))
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(nrows=2,sharex=True)

    ax1.get_position()
    ax1.set_position([0.125,0.37, 0.775, 0.51])
    ax2.set_position([0.125, 0.11, 0.775, 0.22])
    
    ax1.grid(True)
    ax2.grid(True)
#    plt.minorticks_on()
#    plt.grid(b=True, which='minor', linestyle=':')
#    fig.tight_layout()
    
    ns1, bins1, patches1 = ax1.hist(
        actuals,
        bins=range(0,7,1),
        align='left',
        histtype=u'step',
        edgecolor='red',
        linewidth=1.2,
        label='Truth'
    )
    ns2, bins2, patches2 = ax1.hist(
        predictions,
        bins=range(0,7,1),
        align='left',
        histtype=u'step',
        edgecolor='blue',
        linewidth=1.2,
        label='ML pred'
    )
    chi2binpre = np.divide((ns2 - ns1)**2, ns1*ns1.sum())
    ratio = np.divide(ns2, ns1)
    if n_tracks is not None:
        ns3, bins3, patches2 = ax1.hist(
            n_tracks,
            bins=range(0,7,1),
            align='left',
            histtype=u'step',
            edgecolor='green',
            linewidth=1.2,
            label='Tracked'
        )
        chi2bintra = np.divide((ns3 - ns1)**2, ns-1*ns1.sum())
    ax1.legend()
    ax1.ticklabel_format(
        axis='y',
        style='sci',
        scilimits=(0, 2)
    )
    ax1.set_axisbelow(True)
    
    x = np.linspace(-1,6,100)
    y = np.ones_like(x)
    ax2.plot(x, y, '-k', linewidth=.795) #where='mid',
    ax2.plot(
        range(0,6,1),
        ratio,
        'o',#drawstyle='steps-mid',
        color='blue') #where='mid',
    if n_tracks is not None:
        ax2.plot(
            range(0,6,1),
            chi2bintra,
            'o',#drawstyle='steps-mid',
            color='green') #where='mid',
    plt.ylim(0, 2)
#    plt.yscale('log')
    
    plt.xlim((-.8,5.8))
    ax1.yaxis.set_label_coords(-0.075,0.5)
    ax2.yaxis.set_label_coords(-0.075,0.5)
    ax1.set_title(prefix + 'Multiplicity')
    ax1.set_ylabel('Events')
    ax2.set_ylabel('Ratio [ML/Truth]')
    ax2.set_xlabel('Hadron Multiplicity')
    plt.savefig('mult'+ suffix +'.pdf')
    plt.close()
    
    chi2 = round(np.divide((ns2 - ns1)**2 , ns1*ns1.sum()).sum(), 4)
    LOGGER.debug("------------------------------------------------------")
    LOGGER.debug('ML Prediction $\\chi^2 \\approx {}$'.format(chi2))
    if n_tracks is not None:
        chi2 = round(np.divide((ns3 - ns1)**2 , ns1*ns1.sum()).sum(), 4)
        LOGGER.debug('Tracks $\\chi^2 \\approx {}$\n'.format(chi2))
    


def wdistplot(Wdist, bins, label, title, ylog = False, xlog = False):
    plt.figure()
    fig, ax = plt.subplots()
    plt.grid(True)
    if ylog:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')
    plt.title('W distribution')
    plt.xlabel('W [GeV]')
    plt.ylabel('Events')
    plt.hist(Wdist, bins=bins, histtype=u'step', linewidth=1.2, label=label)
    ax.set_axisbelow(True)
    plt.legend()
    if not ylog or not xlog:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
    plt.savefig(title)
    plt.close()



def matplt(actuals, predictions, W, Q2, n_tracks=None, probs=None, DIS=None):
    actuals = actuals.astype(np.int)
    predictions = predictions.astype(np.int)
#    print(W)
##    evts = 2000000
##     
#    probs = probs[:evts]
#    actuals = actuals[:evts]
#    predictions = predictions[:evts]
#    W = W[:evts]
#    Q2 = Q2[:evts]

#----------------------------------cuts----------------------------------------
    
    if DIS == "DIS":
        DoLimQ2 = 1.
    else:
        DoLimQ2 = 0.
    UpLimW      = 20. # 2*np.log(15.0)
    DoLimW      = 0.9
    low_prob    = 0.

    init_size = W.shape[0]
    LOGGER.info('')
    if DoLimQ2 == 1.:
        LOGGER.info('DIS Sample\n')
    else:
        LOGGER.info('Non DIS Sample\n')
    
    if probs is not None:
        predictions = predictions[ probs>low_prob ]
        actuals     = actuals[ probs>low_prob ]
        Q2          = Q2[ probs>low_prob ]
        W           = W[ probs>low_prob ]
    
    actuals     = actuals[Q2>DoLimQ2]
    predictions = predictions[Q2>DoLimQ2]
    if n_tracks is not None:
        n_tracks    = n_tracks[Q2>DoLimQ2]
    W           = W[Q2>DoLimQ2]

    actuals     = actuals[W<UpLimW]
    predictions = predictions[W<UpLimW]
    if n_tracks is not None:
        n_tracks    = n_tracks[W<UpLimW]
    W           = W[W<UpLimW]

    actuals     = actuals[W>DoLimW]    
    predictions = predictions[W>DoLimW]
    if n_tracks is not None:
        n_tracks    = n_tracks[W>DoLimW]
    W           = W[W>DoLimW]
    
    UpLimW = W.max() # 2*np.log(15.0)
    DoLimW = W.min()

    LOGGER.info("There are {} events".format(W.shape[0]))
        
    # 0.9 < W < 1.4 GeV
    actuals0     = actuals[ W<1.4 ]
    predictions0 = predictions[ W<1.4 ]
    if n_tracks is not None:
        n_tracks0    = n_tracks[ W<1.4 ]
    
    # 1.4 < W < 2 GeV
    actuals1     = actuals[W<2]
    actuals1     = actuals1[W[W<2]>1.4]
    predictions1 = predictions[W<2]
    predictions1 = predictions1[W[W<2]>1.4]
    if n_tracks is not None:
        n_tracks1    = n_tracks[W<2]
        n_tracks1    = n_tracks1[W[W<2]>1.4]

    # W > 2 GeV
    actuals2     = actuals[W>2]
    predictions2 = predictions[W>2]
    if n_tracks is not None:
        n_tracks2    = n_tracks[W>2]
#    
    # W multiplicities distribution
    W0A = W[actuals == 0]
    W1A = W[actuals == 1]
    W2A = W[actuals == 2]   
    W3A = W[actuals == 3]
    W4A = W[actuals == 4]
    W5A = W[actuals == 5]
    W0P = W[predictions == 0]
    W1P = W[predictions == 1]
    W2P = W[predictions == 2]
    W3P = W[predictions == 3]
    W4P = W[predictions == 4]
    W5P = W[predictions == 5]
    if n_tracks is not None:
        W0T = W[n_tracks == 0]
        W1T = W[n_tracks == 1]
        W2T = W[n_tracks == 2]
        W3T = W[n_tracks == 3]
        W4T = W[n_tracks == 4]
        W5T = W[n_tracks == 5]
    
    # W dist mean per multiplicity
    WmeanA = np.array([
               (W0A).mean(), (W1A).mean(),
               (W2A).mean(), (W3A).mean(),
               (W4A).mean(), (W5A).mean()
             ])
    WmeanP = np.array([
               (W0P).mean(), (W1P).mean(),
               (W2P).mean(), (W3P).mean(),
               (W4P).mean(), (W5P).mean()
             ])
    if n_tracks is not None:
        WmeanT = np.array([
                   (W0T).mean(), (W1T).mean(),
                   (W2T).mean(), (W3T).mean(),
                   (W4T).mean(), (W5T).mean()
                 ])
    
    WmeanA = 2*np.log(WmeanA)
    WmeanP = 2*np.log(WmeanP)
    if n_tracks is not None:
        WmeanT = 2*np.log(WmeanT)
    
    LOGGER.debug('Actuals values {}'.format(WmeanA))
    LOGGER.debug('ML predtions values {}'.format(WmeanP))
    if n_tracks is not None:
        LOGGER.info('\nTrack based values'.format(Wmeant))
    
    mults = range(0,6,1)
    fitA = np.polyfit(mults, WmeanA, 1)
    fit_fnA = np.poly1d(fitA)
    fitP = np.polyfit(mults, WmeanP, 1)
    fit_fnP = np.poly1d(fitP)
    LOGGER.debug('Actuals fit: {}'.format(fit_fnA))
    LOGGER.debug('ML predtions fit: {}'.format(fit_fnP))
    
    bins = np.logspace(np.log10(DoLimW-0.1*DoLimW),np.log10(UpLimW), 75)
#    bins = np.linspace(DoLimW,UpLimW, 75)
    WP = (W, W0P, W1P, W2P, W3P, W4P, W5P)
    labelP = ('All', 'ML pred 0', 'ML pred 1',
        'ML pred 2', 'ML pred 3', 'ML pred 4', 'ML pred 5')
    
    WA = (W, W0A, W1A, W2A, W3A, W4A, W5A)
    labelA = ('All', 'Truth 0', 'Truth 1', 'Truth 2',
        'Truth 3', 'Truth 4', 'Truth 5')
    
    if n_tracks is not None:
        WT = (W, W0T, W1T, W2T, W3T, W4T, W5T)
        labelT = ('All', 'Tracked 0', 'Tracked 1', 'Tracked 2',
            'Tracked 3', 'Tracked 4', 'Tracked 5')
            
    final_size = W.shape[0]
    lost_size = float(init_size - final_size)
    
    LOGGER.info(
        "Ratio lost events {}\n".format((lost_size)/float(init_size)))
#----------------------------------plots---------------------------------------
    conf_mat(actuals, predictions)
    conf_mat(actuals0, predictions0, '0')
    conf_mat(actuals1, predictions1, '1')
    conf_mat(actuals2, predictions2, '2')

#    conf_mat(actuals, n_tracks, 'tr')
#    conf_mat(actuals0, n_tracks0, 'tr0')
#    conf_mat(actuals1, n_tracks1, 'tr1')
#    conf_mat(actuals2, n_tracks2, 'tr2')

    if n_tracks is not None:
        mult_plot(actuals, predictions, n_tracks)
        mult_plot(actuals0, predictions0, n_tracks0, '0')
        mult_plot(actuals1, predictions1, n_tracks1, '1')
        mult_plot(actuals2, predictions2, n_tracks2, '2')
    else:
        mult_plot(actuals, predictions)
        mult_plot(actuals0, predictions0, suffix='0')
        mult_plot(actuals1, predictions1, suffix='1')
        mult_plot(actuals2, predictions2, suffix='2')
 
    wdistplot(WP, bins, labelP, title="WPredDist.pdf", ylog = True, xlog=True)
    wdistplot(WA, bins, labelA, title="WTruthDist.pdf", ylog = True, xlog=True)
    if n_tracks is not None:
        wdistplot(WT, bins, labelT, title="WTrackedDist.pdf", ylog = True, xlog=True)  
#    
    plt.figure()
    plt.grid(True)
    plt.title(r'Multiplicity as function of $\langle W\rangle^2$')
    plt.xlabel(r'$\langle W \rangle^2$ [GeV]')
    plt.ylabel('Multiplicity')
#    plt.plot( fit_fnA(mults), mults, 'r--', label='Truth Fit ' )
#    plt.plot( fit_fnP(mults), mults, 'b--', label='ML Pred Fit' )
    plt.plot( WmeanA, mults, '^', color='red', label='Truth' )
    plt.plot( WmeanP, mults, 'o', color='blue', label='ML pred' )
    if n_tracks is not None:
        plt.plot( WmeanT, mults, 's', color='green', label='Tracked' )
    plt.legend()
#    plt.xscale('log')
    plt.savefig('MultMeanW.pdf')
    plt.close()





if __name__ == '__main__':
    try:
        # Get prediction path as a parameter
        if len(sys.argv) == 2:
            filename = sys.argv[1]
            DIS = ''
        elif len(sys.argv) == 3:
            filename = sys.argv[1]
            DIS = sys.argv[2]
        else:
            usage()
        
        cwd = os.getcwd()
        basedir =  filename.split('/')[-2]
        playlist = filename.split('_')[-1]
        playlist = playlist.split('.')[0]
        stepmodel =  filename.split('_')[-2]
        
        plotdir = os.path.join(cwd, "plots")
        basedir = os.path.join(plotdir, basedir)
        pldir = os.path.join(basedir, playlist)
        logdir = os.path.join(pldir, stepmodel)
        logdir += DIS
        
        for directory in [plotdir, basedir, pldir, logdir]:
            if not os.path.exists(directory):
                print('{} will be created.'.format(directory))
                os.mkdir(directory)
            else:
                print('{} will be used.'.format(directory))

        logfilename = os.path.join(
            logdir, filename.split('/')[-1].split('.')[0] + DIS + '.log')
        if os.path.exists(logfilename):
            os.remove(logfilename)
            print('{} removed.'.format(logfilename))
        print('{} will be created.'.format(logfilename))



        formatter = logging.Formatter('%(message)s')
            
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(logfilename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        LOGGER = logging.getLogger(__name__)
        LOGGER.setLevel(logging.DEBUG)
        
        LOGGER.addHandler(file_handler)
        LOGGER.addHandler(stream_handler)

        os.chdir(logdir)
        LOGGER.info("Playlist: {}".format(playlist))
        hdf5 = '/lfstev/e-938/jbonilla/hdf5/{}/hadmultkineimgs_127x94_{}-part1.hdf5'.format(
          playlist, playlist)
        reader = HDF5Reader(hdf5)
        nevents = reader.openf()

        index_dict = {
            'predictions': 4
        }
        data_dict = file2npy(filename, index_dict) #, W, Q2, n_tracks actuals,
        nevents =  data_dict['predictions'].shape[0]
        actuals = reader.get_key(0,nevents,key='hadro_data/n_hadmultmeas_50mev')

        W = reader.get_key(0, nevents, key='gen_data/W')/1000
        Q2 = reader.get_key(0, nevents, key='gen_data/Q2')/1000000
#        n_tracks[ n_tracks>5 ] = 5
        n_tracks = None
        matplt(actuals, data_dict['predictions'], W, Q2, n_tracks, DIS=DIS)
        os.chdir(cwd)

    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
    
