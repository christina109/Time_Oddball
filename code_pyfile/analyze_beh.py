# correspondence: cyj.sci@gmail.com

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

p_main = os.getcwd()
f_input = os.path.join(p_main, 'beh')
f_output = os.path.join(p_main, 'temp_data')
f_graph = os.path.join(p_main, 'temp_graph')

# test only
sub = 101
conds = ['grating_freq', 'direction', 'stim_type']
conds_val = [20, 'increament', 'oddball']
conds_vals = [[0,5,10,20],['decreament', 'increament'],['standard','oddball']]
beh_indices = ['resp.corr', 'resp.rt']


def combine_conds_val(conds_vals):
    nCond = len(conds_vals)
    tp = 'product('
    for ci in range(nCond):
        tp += 'conds_vals[{}],'.format(ci)
    tp = tp[:-1]
    tp+=')'
    ps = eval(tp)

    conds_val_list = []
    for p in ps:
        conds_val_list.append(list(p))

    return conds_val_list


def format_f_output(f_output):
    if f_output[-4:] == '.csv':
        return f_output
    else:
        return '{}.csv'.format(f_output)


def safesave(df):
    folder_output = input('Output file name: ')
    path = os.path.join(f_output, format_f_output(folder_output))
    saveOn = True
    while os.path.exists(path):
        resp = input('File EXISTS. Overwrite?[y/n/q(quit)]')
        if resp == 'y' or resp == 'Y':
            break
        elif resp == 'q' or resp == 'Q':
            saveOn = False
            print('Abort')
            break
        else:
            folder_output = input('Output file name: ')
            path = os.path.join(f_output, format_f_output(folder_output))
    if saveOn:
        df.to_csv(path)
        print('Saved to {}'.format(path))


def detect_outliers(arr):  # outliers are marked as nan
    arr = pd.DataFrame(arr)
    q75, q25 = np.percentile(arr, [75, 25])
    intr_qr = q75 - q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    arr[arr< min] = np.nan
    arr[arr> max] = np.nan

    return arr


def detect_outliers_by_cond(df, y, conds, conds_vals):
    val_comb_list = combine_conds_val(conds_vals)
    subs_rm_all_conds = []
    df_new = None
    for val_comb in val_comb_list:
        tp = df
        for cond, cond_val in zip(conds, val_comb):
            tp = tp[tp[cond]==cond_val]
        tp[y] = detect_outliers(tp[y])

        if df_new is None:
            df_new = tp
        else:
            df_new = pd.concat([df_new, tp])
        subs_rm = tp[tp[y].isnull()]['participant']
        subs_rm = subs_rm.tolist()
        subs_rm_all_conds.extend(subs_rm)
        print('Outlier dataset in {} {} is {}'.format(conds, val_comb, subs_rm))
    print('***')
    subs_rm_all_conds = np.unique(subs_rm_all_conds)
    print('{} datasets have outliers in at least one condition: {}'.format(len(subs_rm_all_conds), subs_rm_all_conds))

    return df_new.dropna()


def plot_reg(df, x, y, cond, cond_vals, group, group_vals):
   #sns.set(rc={'figure.figsize':(15,5)})

    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    #sns.set(font_scale=2)
    sns.set_style('white')
    fname_save = input('Output file name: ')

    print('Working on the plotting...')
    sns.lmplot(x=x,y=y,data=df,
               hue=cond,col=group,
               x_estimator = np.mean,
               palette = 'viridis', height = 5, aspect = 0.8)
    plt.savefig(os.path.join(f_graph, '{}_normal.png'.format(fname_save)),
                dpi = 300)

    sns.lmplot(x=x,y=y,data=df,
               hue=cond,col=group,
               x_estimator = np.mean, logx=True,
               palette = 'viridis', height = 5, aspect = 0.8)
    plt.savefig(os.path.join(f_graph, '{}_logx.png'.format(fname_save)),
                dpi = 300)
    print('Graphs saved to {}'.format(f_graph))


def fit_reg(df, x, y, conds, conds_vals):
    X = df[conds]
    X2 = sm.add_constant(X)
    est = sm.OLS(np.array(df[y]), np.array(X2))
    est2 = est.fit()
    est2.summary()


# behavioral data per condition
class beh_cond():
    task = 'oddball'

    def __init__(self, sub, conds, conds_val,session=None):
        self.sub = sub
        self.conds = conds
        self.conds_val = conds_val
        data = pd.read_csv(os.path.join(f_input, 'beh_{}_oddball.csv'.format(self.sub)))
        if session == None:
            self.data = data
        else:
            self.data =  data.loc[data['blocks.thisN']==session,:]

        for cond, cond_val in zip(conds, conds_val):  # inner join
            self.data = self.data.loc[self.data[cond]==cond_val,:]

    def calc_stats(self, beh_indices):
        self.mean = []
        self.sd = []
        self.n = []
        self.beh_indices = beh_indices

        for beh_index in beh_indices:
            if self.data.shape[0] > 0:
                self.mean.append(self.data[beh_index].mean())
                self.sd.append(self.data[beh_index].std())
            else:
                self.mean.append(None)
                self.sd.append(None)
        self.n.append(self.data.shape[0])

    def get_mean(self, beh_indices):
        self.calc_stats(beh_indices)
        return self.mean


# quick summary of acc/rt between oddball/standard
class beh_sub_sum:
    beh_indices = ['resp.corr', 'resp.rt']

    def __init__(self, sub):
        self.sub = sub
        self.beh_indices = beh_indices
        self.mean = {}
        for stim_type in ['standard', 'oddball']:
            tp = beh_cond(sub, ['stim_type'], [stim_type]).get_mean(beh_indices)
            self.mean.update({stim_type: tp})


class beh_group_session():
    def __init__(self, subs, conds=conds, conds_vals=conds_vals, beh_indices=beh_indices, session=0):
        self.subs = subs
        self.conds = conds
        self.conds_vals = conds_vals
        self.beh_indices = beh_indices
        self.session = session

        conds_val_list = combine_conds_val(conds_vals)

        self.data = [] # *cheaper* to append a list than df
        print('Loading data...')
        for sub in tqdm(subs):
            for conds_val in conds_val_list:
                line = [sub]
                line.extend(conds_val)
                b = beh_cond(sub, conds, conds_val, session)
                line.extend(b.get_mean(beh_indices))
                line.extend(b.n)
                self.data.append(line)
        self.data = pd.DataFrame(self.data, columns=['sub'] + conds + beh_indices + ['n'])


class beh_group:
    def __init__(self, subs, conds=conds, conds_vals=conds_vals, beh_indices=beh_indices, exportOn=False):
        self.subs = subs
        self.conds = conds
        self.conds_vals = conds_vals
        self.beh_indices = beh_indices

        conds_val_list = combine_conds_val(conds_vals)

        self.data = []
        print('Loading data...')
        for sub in tqdm(subs):
            for conds_val in conds_val_list:
                line = [sub]
                line.extend(conds_val)
                b = beh_cond(sub, conds, conds_val)
                line.extend(b.get_mean(beh_indices))
                line.extend(b.n)
                self.data.append(line)

        self.data = pd.DataFrame(self.data, columns = ['sub']+conds+beh_indices+['n'])
        if exportOn:
            safesave(self.data)

    def plot_beh(self, beh_index='resp.corr', cond_x='grating_freq', cond_group=None, cond_subplot=None):
        if cond_subplot is None:
            nSubplot = 1
        else:
            conds_subplot = self.data[cond_subplot].unique()
            nSubplot = len(conds_subplot)

        fig, axes = plt.subplots(1, nSubplot, sharex=True, sharey=True)

        for pi in range(nSubplot):
            if cond_subplot is None:
                tp = self.data
            else:
                tp = self.data[self.data[cond_subplot] == conds_subplot[pi]]

            #if cond_x == 'scaling' and beh_index == 'resp.corr':
                #tp.loc[tp[cond_x]==100,'resp.corr'] = 1-tp.loc[tp[cond_x]==100,'resp.corr']

            if cond_group is None:
                sns.boxplot(ax=axes if cond_subplot is None else axes[pi],
                            x=cond_x, y=beh_index, data=tp, palette = 'viridis')
            else:
                sns.boxplot(ax=axes if cond_subplot is None else axes[pi],
                            x=cond_x, y=beh_index, hue = cond_group, data=tp, palette = 'viridis')

            if not cond_subplot is None:
                if cond_subplot == 'grating_freq':
                    axes[pi].set_title('{}Hz'.format(conds_subplot[pi]))
                else:
                    axes[pi].set_title(conds_subplot[pi])

        plt.show()

    def calc_beh_sum(self):

        if 'stim_type' not in self.conds:
            return

        self.acc_o = []; self.acc_s = []
        self.rt_o = []; self.rt_s = []
        for sub in self.subs:
            s = beh_sub_sum(sub)
            self.acc_o.append(s.mean['oddball'][0])
            self.rt_o.append(s.mean['oddball'][1])
            self.acc_s.append(s.mean['standard'][0])
            self.rt_s.append(s.mean['standard'][1])

        np.savetxt(os.path.join(f_output, 'acc_o.csv'), self.acc_o, delimiter = ',')
        np.savetxt(os.path.join(f_output, 'acc_s.csv'), self.acc_s, delimiter=',')
        np.savetxt(os.path.join(f_output, 'rt_o.csv'), self.rt_o, delimiter=',')
        np.savetxt(os.path.join(f_output, 'rt_s.csv'), self.rt_s, delimiter=',')
        print('Summarised data are saved to folder: {}'.format(f_output))

    def plot_beh_sum(self):
        self.calc_beh_sum()
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot( self.subs, self.acc_o, '-ro')
        axes[0].plot( self.subs, self.acc_s, '-b^')
        axes[0].set_ylabel('Accuray')
        axes[0].legend()

        axes[1].plot( self.subs, self.rt_o, '--ro')
        axes[1].plot( self.subs, self.rt_s, '--b^')
        axes[1].set_ylabel('Reponse Time [s]')
        plt.show()

    def calc_dprime(self, cond, cond_vals, cond_group=None, group_val=None):
        if 'stim_type' not in self.conds:
            print('Please include STIM_TYPE as a feature')
            return
        if 'resp.corr' not in self.beh_indices:
            print('Please include RESP.CORR in the data')
            return

        if cond == 'scaling' and 'direction' not in self.conds:
            print('Please include DIRECTION as a feature')

        if cond_group is not None:
            tp_all = self.data.loc[self.data[cond_group] == group_val,:]
        else:
            tp_all = self.data.copy()
        for sub in self.subs:
            tp_sub = tp_all.loc[tp_all['sub'] == sub,:]
            tp_sub = tp_sub.dropna()
            d_arr = []
            for cond_val in cond_vals:
                tp = tp_sub.loc[tp_sub[cond] == cond_val,:]
                if not cond == 'scaling':
                    hit = tp.loc[tp['stim_type'] == 'oddball', 'resp.corr']
                    n_pos = tp.loc[tp['stim_type'] == 'oddball', 'n']
                    fa = 1 - tp.loc[tp['stim_type'] == 'standard', 'resp.corr']
                    n_neg = tp.loc[tp['stim_type'] == 'standard', 'n']
                else:
                    hit = tp['resp.corr']
                    n_pos = tp['n']
                    tp2 = tp_sub.loc[tp_sub['stim_type'] == 'standard',:]
                    if cond_val in [40,60,80]:
                        tp2 = tp2.loc[tp2['direction'] == 'decreament', :]
                    elif cond_val in [120,140,160]:
                        tp2 = tp2.loc[tp2['direction'] == 'increament', :]
                    fa = 1-tp2.loc[:, 'resp.corr']
                    n_neg = tp2.loc[:, 'n']

                # correct to avoid 0/1
                if cond == 'scaling':
                    hit = (n_pos*hit+0.1/3)/(n_pos+0.2/3)
                    fa = (n_neg*fa+0.9)/(n_neg+1.8)
                else:
                    hit = (n_pos*hit+0.1)/(n_pos+0.2)
                    fa = (n_neg*fa+0.9)/(n_neg+1.8)

                d = norm.ppf(hit) - norm.ppf(fa)
                d_arr.extend(d)

            df = pd.DataFrame.from_dict({'participant': [sub] * len(d_arr),
                                         cond: cond_vals,
                                         'd': d_arr})
            if sub == self.subs[0]:
                df_n = df
            else:
                df_n = pd.concat([df_n, df])
        return df_n

    def plot_dprime(self, cond_x, x_vals, cond_group=None, group_vals=None):

        if cond_group is None:
            group_vals = ['None'] # virtual group_vals for the loop

        fig, axes = plt.subplots(1, len(group_vals), sharex=True, sharey=True)
        for gi in range(len(group_vals)):
            if cond_group is None:
                ax = axes
            else:
                ax = axes[gi]
            gv = group_vals[gi]
            df = self.calc_dprime(cond_x, x_vals, cond_group, gv)

            sns.boxplot(ax=ax,
                        x = cond_x, y = 'd',
                        data=df, palette = 'viridis')
            ax.set(xlabel = cond_x)
            if cond_x == 'scaling':
                ax.set_xticklabels(np.array(x_vals)-100)

            if cond_group is not None:
                ax.set_title(group_vals[gi])

            df[cond_group] = gv
            if gi == 0:
                self.d_group = df
            else:
                self.d_group = pd.concat([self.d_group, df])
            ax.set(ylabel = '')
        if cond_group is None:
            axes.set(ylabel = 'd\'')
        else:
            axes[0].set(ylabel = 'd\'')

        plt.show()













