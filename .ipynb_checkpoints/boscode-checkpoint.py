import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import glob
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import cm
from skimage.transform import resize
import mpld3
import matplotlib.ticker as tkr 
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def load_image(core_label, core_section = -1):
    
    global core, core_images, images_shp
    
    if core_section > 0:
        
        file = f'data/images/JC036_{core_label}_{core_section}/IM*'
        files = sorted(glob.glob(file))
        
    else:

        file = f'data/images/JC036_{core_label}_*/IM*'
        files = sorted(glob.glob(file))

    rgb = [i for i in files if 'rgb' in i]
    photos = [i for i in files if 'tif' in i]
    photos = [i for i in photos if not '_R' in i]

    appended_data = []
    images = []
    images_shp = []

    for i, k in zip(rgb, photos):

        df = pd.read_csv(i, header = 7, delimiter = '\t')
        df.reset_index(inplace = True)
        df.rename(columns={'level_0': 'height', 'level_1': 'r', 'level_2': 'g', 'Data:': 'b'}, inplace=True)
        df['rgb'] = df.r + df.g + df.b
        df = df[df.height < (df.height.max() - 6.5)]
        df['height'] = df['height']/100
        df.reset_index(inplace = True)
        appended_data.append(df)
        img = plt.imread(k)
        img = img[:-1300]
        images_shp.append(img.shape[:2])
        images.append(img)

    core_images = np.vstack(images)
    core = pd.concat(appended_data)
    core.reset_index(inplace = True)
    core['depth_m'] = core.index/1000
    
    print('image: done')
    
    return core, core_images, images_shp

def load_xrf(core_label = None, core_section = -1, path = None):
    
    global xrf
    
    if path != None:
    
        file = f'{path}'
        files = sorted(glob.glob(file))
        files = [i for i in files if not '_REP' in i]
        
    if core_section > 0:
        
        file = f'data/itrax/JC36/JC36_{core_label}_{core_section}/JC36_{core_label}_{core_section}/Results.txt'
        files = sorted(glob.glob(file))
        files = [i for i in files if not '_REP' in i]
        
    if core_label != None:

        file = f'data/itrax/JC36/JC36_{core_label}_*/JC36_{core_label}_*/Results.txt'
        files = sorted(glob.glob(file))
        files = [i for i in files if not '_REP' in i]

    appended_data = []
    prev_position = []

    for j in files:

        df = pd.read_csv(j, header = 1, delimiter = '\t')
        df.reset_index(inplace = True)
        df.columns = df.iloc[0]
        df.drop(df.index[0], inplace = True)
        df[['position (mm)', 'validity', 'MSE','Al','Si','P','S','Cl','Ar','K','Ca','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','As','Br','Rb','Sr','Y','Zr','Sb','Ba','Ce','Sm','Ta','W','Pb']
        ] = df[['position (mm)', 'validity', 'MSE','Al','Si','P','S','Cl','Ar','K','Ca','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','As','Br','Rb','Sr','Y','Zr','Sb','Ba','Ce','Sm','Ta','W','Pb']
        ].apply(pd.to_numeric)

        if len(prev_position) == 0:

            prev_position.append(df['position (mm)'].max()) 
            df['position_corr'] = df['position (mm)']

        else:

            df['position_corr'] = (df['position (mm)'] + max(prev_position)) - df['position (mm)'].min()
#             df['position_corr'] = (df['position (mm)'] + max(prev_position))

            prev_position.append(df['position_corr'].max())  

#         df = df[df.validity == 1]

        appended_data.append(df)
        
    xrf = pd.concat(appended_data)
    xrf.reset_index(inplace = True)
    
    xrf_pair = pd.read_csv('data/cluster/xrf_pair.csv')
    xrf_pair = xrf_pair.drop(columns = 'filename')
    xrf = pd.merge(xrf, xrf_pair, how = 'left', on = 'Si')
    xrf['color'] = ['gold' if i == 0 else
                'tan' if i == 2 else
                'grey' if i == 1 else
                None for i in xrf['clusters']]
    xrf[['position_corr', 'Ca', 'Si', 'Fe', 'Sr', 'Ti','clusters']].to_csv(f'data/cluster/{core_label}_clusters.csv')

    print('xrf: done')
    
    return xrf


def load_xray(core_label, core_section = -1):
    
    global xrays, lams
    
    if core_section > 0:
        
        xray_file = f'data/scoutxscan/JC36-{core_label}-{core_section}/Radiographs/*.TIF'
        xray_files = sorted(glob.glob(xray_file))

        lam_file = f'data/scoutxscan/JC36-{core_label}-{core_section}/Laminography slices/*+45*.tif'
        lam_files = sorted(glob.glob(lam_file))
        
    else:
            
        xray_file = f'data/scoutxscan/JC36-{core_label}-*/Radiographs/*.TIF'
        xray_files = sorted(glob.glob(xray_file))

        lam_file = f'data/scoutxscan/JC36-{core_label}-*/Laminography slices/*+45*.tif'
        lam_files = sorted(glob.glob(lam_file))

    xray_images = []
    lam_images = []

    counter = 0
    
    for i, k in zip(xray_files, lam_files):

        img = plt.imread(i)
        img = resize(img, images_shp[counter], anti_aliasing = True)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        xray_images.append(img)

        img = plt.imread(k)
        img = resize(img, images_shp[counter], anti_aliasing = True)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        lam_images.append(img)

        counter += 1

    xrays = np.vstack(xray_images)
    lams = np.vstack(lam_images)
    
    print('xray: done')
        
    return xrays, lams

def load_magsus(core_label, core_section = -1):
    
    global magsus
    
    mag = pd.read_csv(f'data/mag_sus/{core_label}_mag_sus.csv', header=None, sep='\n')
    mag = mag[0].str.split(',', expand=True)
    mag = mag.iloc[3:]

    left = mag.iloc[:, :10]
    right = mag.iloc[:, 10:]

    right.drop(3, inplace = True)
    right.columns = right.iloc[0]
    right.drop(4, inplace = True)

    left.drop(4, inplace = True)
    left.columns = left.iloc[0]
    left.drop(3, inplace = True)

    magsus = pd.concat([left, right], axis=1)
    magsus = magsus.apply(pd.to_numeric,  errors='ignore')
    
    print('magsus: done')

    return magsus

def plot_all(core_label, core_section = -1):
    
    fig, ax = plt.subplots(ncols = 9, figsize = (10, 8))
    # fig, ax = plt.subplots(ncols = 9)

    left_lim = 2
    right_lim = 8

    span = abs(left_lim - right_lim)
    cmap = plt.get_cmap('Greys')

    color_index = np.arange(left_lim, right_lim, span / 100)
    
    ax[0].imshow(core_images)
    ax[0].set(title = 'image')
    ax[0].set_aspect('auto')
    ax[0].set(ylim = (core_images.shape[0], 0), yticklabels = np.arange(0, core_images.shape[0]/20000, 1),
             ylabel = 'depth [m]')
    ax[0].tick_params(axis='y', which='minor', bottom=False)
    ax[0].vlines(core_images.shape[1]/2, core_images.shape[0], 0, color = 'k', linestyle = '--') 
    ax[0].axes.get_xaxis().set_visible(False)

    ax[1].plot(core.r, core.depth_m, c = 'k', lw = 1)
    ax[1].set(ylim = (core.depth_m.max(), core.depth_m.min()), xlim = (12, 2), ylabel = 'depth [m]',
             xlabel = 'r-intensity', title = 'r-intensity')
    ax[1].fill_betweenx(core.depth_m, core.r, 0, color = 'w')
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)

    for index in sorted(color_index):

        index_value = (index - left_lim)/span
        color = cmap(index_value)
        ax[1].fill_betweenx(core. depth_m, 12, core.r, where = core.r >= index,  color = color)

    cax = ax[2].imshow(xrays, cmap = 'Greys_r', vmin = 0., vmax = 0.25)
    ax[2].set_aspect('auto')
    ax[2].axes.get_xaxis().set_visible(False)
    ax[2].axes.get_yaxis().set_visible(False)
    ax[2].set(title = 'x-ray')

#     ax[3].imshow(lams, cmap = 'Greys_r', vmin = 0.3, vmax = 0.8)
#     ax[3].set_aspect('auto')
#     ax[3].axes.get_xaxis().set_visible(False)
#     ax[3].axes.get_yaxis().set_visible(False)
#     ax[3].set(title = 'x-ray')
# #     ax[3].set(xlim = (600, 1500))

    if core_section > 0:

        xrf_depth = xrf['position (mm)']/1000
        
    else: 
        
        xrf_depth = xrf.position_corr/1000

    ax[3].plot(xrf['Si']/xrf['Ca'], xrf_depth, c = 'k', lw = 0.7)
    ax[3].set(ylim = (core.depth_m.max(), core.depth_m.min()), title = 'Si/Ca')
    ax[3].axes.get_yaxis().set_visible(False)

    ax[4].plot(xrf['Ca']/xrf['Ti'], xrf_depth, c = 'k', lw = 0.8)
    ax[4].set(ylim = (core.depth_m.max(), core.depth_m.min()), title = 'Ca/Ti')
    ax[4].axes.get_yaxis().set_visible(False)

    ax[5].plot(xrf['Sr']/xrf['Ca'], xrf_depth, c = 'k', lw = 0.8)
    ax[5].set(ylim = (core.depth_m.max(), core.depth_m.min()), xlim = (0.05, 0.15), title = 'Sr/Ca')
    ax[5].axes.get_yaxis().set_visible(False)

    ax[6].plot(xrf['Br']/xrf['Cl'], xrf_depth, c = 'k', lw = 0.8)
    ax[6].set(ylim = (core.depth_m.max(), core.depth_m.min()), xlim = (0, 0.7), title = 'Br/Cl')
    ax[6].axes.get_yaxis().set_visible(False)

    xrf['next'] = xrf['position (mm)'].shift(-1)
    xrf['next_color'] = xrf.color.shift(-1)

    thickness = []
    index = []
    clrs = []
    counter = 0.02

    for layer in zip(xrf.position_corr/1000, xrf.next/1000, xrf.color, xrf.next_color, xrf.index):

        if layer[2] == layer[3]:

            counter += 0.002

        else:

            thickness.append(counter)
            index.append(layer[4])
            clrs.append(layer[2])
            counter = 0.002

        ax[7].axhspan(layer[0], layer[1], 0, 3, color = layer[2])

    thickness_df = pd.DataFrame({'thickness':thickness, 'colors':clrs}, index = index)

    ax[7].axes.get_xaxis().set_visible(False)
    ax[7].set(ylim = (core.depth_m.max(), core.depth_m.min()), title = 'clusters')
    ax[7].axes.get_xaxis().set_visible(False)
    ax[7].axes.get_yaxis().set_visible(False)
    
    ax[8].plot(magsus['Magnetic Susceptibility'], magsus['Depth'], c = 'k', lw = 0.7)
    ax[8].set(ylim = (core.depth_m.max()*100, core.depth_m.min()*100), title = 'mag. sus.')
    ax[8].axes.get_yaxis().set_visible(False)

    for ax in ax:

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()
    
#     plt.savefig(f'figs/{core_label}.jpg', dpi = 400)
    
    print('plot: done')


def plot_zoom(core_label, base, top, c1, c2):
    
#     fig, ax = plt.subplots(ncols = 3, figsize = (10, 8))
    fig, ax = plt.subplots(ncols = 3)

    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
#     ax4 = ax4

    left_lim = 2
    right_lim = 8

    span = abs(left_lim - right_lim)
    cmap = plt.get_cmap('Greys')

    color_index = np.arange(left_lim, right_lim, span / 100)
    
#     xrf['position_corr'] = xrf['position_corr'] - 12
    
    ax1.imshow(core_images)
    ax1.set(title = 'image')
    ax1.set_aspect('auto')
    ax1.set(ylim = (top * 20000, base * 20000), ylabel = 'depth [m]')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.tick_params(axis='y', which='minor', bottom=False)
    ax1.vlines(core_images.shape[1]/2, top * 20000, base * 20000, color = 'k', linestyle = '--') 

    def numfmt(x, pos): 
        
        s = '{}'.format(x / 20000)
        
        return s

    yfmt = tkr.FuncFormatter(numfmt) 

    ax1.yaxis.set_major_formatter(yfmt)
    
#     ax2.imshow(xrays, cmap = 'Greys_r', vmin = 0., vmax = 0.25)
#     ax2.set(ylim = (ylim2 * 20000, ylim1 * 20000))
#     ax2.set_aspect('auto')
#     ax2.axes.get_xaxis().set_visible(False)
#     ax2.axes.get_yaxis().set_visible(False)
#     ax2.set(title = 'x-ray')
    
    ax2.imshow(lams, cmap = 'Greys_r', vmin = c1, vmax = c2)
    ax2.set(ylim = (top * 20000, base * 20000))
    ax2.set_aspect('auto')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set(title = 'x-ray [lam]')
    ax2.axvspan(0, 500, color = 'w')
    ax2.axvspan(lams.shape[1] - 500, lams.shape[1], color = 'w')

    ax3.plot(xrf['Si']/xrf['Ca'], xrf.position_corr/1000, c = 'k', lw = 0.8)
    ax3.set(ylim = (top, base), title = 'Si/Ca')
    ax3.axes.get_yaxis().set_visible(False)

    for axes in [ax1, ax2, ax3]:

        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)
        
    plt.tight_layout(w_pad = 0) 
    
    print('plot: done')
