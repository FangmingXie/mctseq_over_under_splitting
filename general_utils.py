"""
Fangming Xie, Chris Keown
Many functions are adopted from Chris's mypy
"""

from __init__ import *

import subprocess as sp
import os
from scipy import sparse

def myScatter(ax, df, x, y, l, 
              s=20,
              sample_frac=None,
              sample_n=None,
              legend_size=None,
              legend_kws=None,
              grey_label='unlabeled',
              shuffle=True,
              random_state=None,
              legend_mode=0,
              kw_colors=False,
              colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9'], **kwargs):
    """
    take an axis object and make a scatter plot

    - kw_colors is a dictinary {label: color}
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    # shuffle (and copy) data
    if sample_n:
        df = (df.groupby(l).apply(lambda x: x.sample(min(len(x), sample_n), random_state=random_state))
                            .reset_index(level=0, drop=True)
            )
    if sample_frac:
        df = (df.groupby(l).apply(lambda x: x.sample(frac=sample_frac, random_state=random_state))
                            .reset_index(level=0, drop=True)
            )
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    if not kw_colors:
        # add a color column
        inds, catgs = pd.factorize(df[l])
        df['c'] = [colors[i%len(colors)] if catgs[i]!=grey_label else 'grey' 
                    for i in inds]
    else:
        df['c'] = [kw_colors[i] if i!=grey_label else 'grey' for i in df[l]]
    
    # take care of legend
    if legend_mode != -1:
        for ind, row in df.groupby(l).first().iterrows():
            ax.scatter(row[x], row[y], c=row['c'], label=ind, s=s, **kwargs)
        
    if legend_mode == -1:
        pass
    elif legend_mode == 0:
        lgnd = ax.legend()
    elif legend_mode == 1:
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        lgnd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              ncol=6, fancybox=False, shadow=False) 
    elif legend_mode == 2:
        # Shrink current axis's width by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0 + box.width*0.1, box.y0,
                                 box.width*0.8, box.height])

    if legend_kws:
        lgnd = ax.legend(**legend_kws)

    if legend_mode != -1 and legend_size:
        for handle in lgnd.legendHandles:
            handle._sizes = [legend_size] 

    # backgroud (grey)
    df_grey = df.loc[df['c']=='grey']
    if not df_grey.empty:
        ax.scatter(df_grey[x], 
                   df_grey[y],
                   c=df_grey['c'], s=s, **kwargs)
    # actual plot
    df_tmp = df.loc[df['c']!='grey']
    ax.scatter(df_tmp[x], 
               df_tmp[y],
               c=df_tmp['c'], s=s, **kwargs)
    
    return

def plot_tsne_labels_ax(df, ax, tx='tsne_x', ty='tsne_y', tc='cluster_ID', 
                    sample_frac=None,
                    sample_n=None,
                    legend_size=None,
                    legend_kws=None,
                    grey_label='unlabeled',
                    legend_mode=0,
                    s=1,
                    shuffle=True,
                    random_state=None,
                    t_xlim='auto', t_ylim='auto', title=None, 
                    legend_loc='lower right',
                    colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9'], **kwargs):
    """
    tSNE plot

    xlim, ylim is set to facilitate displaying glial clusters only

    # avoid gray-like 'C7' in colors
    # color orders are arranged for exci-inhi-glia plot 11/1/2017
    """
    import matplotlib.pyplot as plt

    myScatter(ax, df, tx, ty, tc,
             s=s,
             sample_frac=sample_frac,
             sample_n=sample_n,
             legend_size=legend_size,
             legend_kws=legend_kws,
             shuffle=shuffle,
             grey_label=grey_label,
             random_state=random_state, 
             legend_mode=legend_mode, 
             colors=colors, **kwargs)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(tc)
    ax.set_xlabel(tx)
    ax.set_ylabel(ty)
    # ax.set_aspect('auto')

    if t_xlim == 'auto':
        t_xlim = [np.nanpercentile(df[tx].values, 0.1), np.nanpercentile(df[tx].values, 99.9)]
        t_xlim[0] = t_xlim[0] - 0.1*(t_xlim[1] - t_xlim[0])
        t_xlim[1] = t_xlim[1] + 0.1*(t_xlim[1] - t_xlim[0])
        ax.set_xlim(t_xlim)
    elif t_xlim:
        ax.set_xlim(t_xlim)
    else:
        pass  

    if t_ylim == 'auto':
        t_ylim = [np.nanpercentile(df[ty].values, 0.1), np.nanpercentile(df[ty].values, 99.9)]
        t_ylim[0] = t_ylim[0] - 0.1*(t_ylim[1] - t_ylim[0])
        t_ylim[1] = t_ylim[1] + 0.1*(t_ylim[1] - t_ylim[0])
        ax.set_ylim(t_ylim)
    elif t_ylim:
        ax.set_ylim(t_ylim)
    else:
        pass

    return

def get_index_from_array(arr, inqs, na_rep=-1):
    """Get index of array
    """
    arr = np.array(arr)
    arr = pd.Series(arr).reset_index().set_index(0)
    idxs = arr.reindex(inqs)['index'].fillna(na_rep).astype(int).values
    return idxs

def nondup_legends(ax='', **kwargs):
    """Assuming plt (matplotlib.pyplot) is imported
    """
    from collections import OrderedDict
    import matplotlib.pyplot as plt

    if ax == '':
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), **kwargs)
    else:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), **kwargs)
    return 

def dedup_array_elements(x, empty_string=''):
    """Replacing repeats with empty_string
    """
    newx = np.empty_like(x)
    newx[0] = x[0]
    for i in range(1, len(x)):
        if x[i-1] == x[i]:
            newx[i] = empty_string
        else:
            newx[i] = x[i]
    return newx

def zscore(x, offset=1e-7, ddof=1):
    return (x - np.mean(x))/(np.std(x, ddof=ddof) + offset)


# smooth-within modality
def smooth_in_modality(counts_matrix, norm_counts_matrix, k, ka, npc=100, sigma=1.0, p=0.1, drop_npc=0):
    """Smooth a data matrix
    
    Arguments:
        - counts_matrix (pandas dataframe, feature by cell)
        - norm_counts_matrix (pandas dataframe, feature by cell) log10(CPM+1)
        - k (number of nearest neighbors)
    Return:
        - smoothed cells_matrix (pandas dataframe)
        - markov affinity matrix
    """
    import fbpca
    import knn_utils
    
    assert counts_matrix.shape[1] == norm_counts_matrix.shape[1] 

    c = norm_counts_matrix.columns.values
    N = len(c)

    # reduce dimension fast version
    U, s, Vt = fbpca.pca(norm_counts_matrix.T.values, k=npc)
    pcs = U.dot(np.diag(s))
    if drop_npc != 0:
        pcs = pcs[:, drop_npc:]

    # get k nearest neighbor distances fast version 
    inds, dists = knn_utils.gen_knn_annoy(pcs, k, form='list', 
                                                metric='euclidean', n_trees=10, search_k=-1, verbose=True, 
                                                include_distances=True)
    
    # remove itself
    dists = dists[:, 1:]
    inds = inds[:, 1:]

    # normalize by ka's distance 
    dists = (dists/(dists[:, ka].reshape(-1, 1)))

    # gaussian kernel
    adjs = np.exp(-((dists**2)/(sigma**2))) 

    # construct a sparse matrix 
    cols = np.ravel(inds)
    rows = np.repeat(np.arange(N), k-1) # remove itself
    vals = np.ravel(adjs)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Symmetrize A (union of connection)
    A = A + A.T

    # normalization fast (A is now a weight matrix excluding itself)
    degrees = A.sum(axis=1)
    A = sparse.diags(1.0/np.ravel(degrees)).dot(A)

    # include itself
    eye = sparse.identity(N)
    A = p*eye + (1-p)*A
    
    # smooth fast (future?)
    counts_matrix_smoothed = pd.DataFrame((A.dot(counts_matrix.T)).T, 
                                         columns=counts_matrix.columns, index=counts_matrix.index)
    return counts_matrix_smoothed, A
