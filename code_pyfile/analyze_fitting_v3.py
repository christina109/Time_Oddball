import matplotlib   # sys.exit()
matplotlib.use('TkAgg')
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from code_pyfile.analyze_beh import beh_group
from scipy.io import savemat, loadmat
import pingouin as pg
from itertools import product
import pickle
import warnings
from time import time

sns.set_theme(style="white", font_scale=2)

def load_channels(task_channels=True):
    if task_channels:
        f_load = 'chanstructure.csv'
    else:
        f_load = 'chanstructure32.csv'
    df = pd.read_csv(f_load)
    return df.labels.tolist()

def get_channel(ci, task_channel=True):
    ci = int(ci)
    channels = load_channels(task_channel)
    return channels[ci]

def load_chanlocs(task_channels=True):
    if task_channels:
        f_load = 'chanstructure.csv'
    else:
        f_load = 'chanstructure32.csv'
    df = pd.read_csv(f_load)
    return df

def get_colors(var_list):
    if type(var_list) == str:
        var_list = [var_list]
    colors = []
    for v in var_list:
        v = v.lower()
        if v == 'hr':
            colors.append('#4169E1')
        elif v == 'cr':
            colors.append('#FF8C00')
        elif v == 'nonadapt' or v =='non-adapt':
            colors.append('#8A2BE2')
        elif v == 'novel' or v == 'oddball':
            colors.append('#008000')
        elif v == 'adapt':
            colors.append('#00FF00')
        elif v == 'rs':
            colors.append('#54939c')
        elif v == 'st':
            colors.append('#d42d2d')
        elif v == 'ap_e':
            colors.append(sns.color_palette('gnuplot2', 7).as_hex()[0])
        elif v == 'ap_o':
            colors.append(sns.color_palette('gnuplot2', 7).as_hex()[1])
        elif v == 'α_f':
            colors.append(sns.color_palette('gnuplot2', 7).as_hex()[2])
        elif v == 'α_pw':
            colors.append(sns.color_palette('gnuplot2', 7).as_hex()[3])
        elif v == 'α_w':
            colors.append(sns.color_palette('gnuplot2', 7).as_hex()[4])
        elif v == 'β_f':
            colors.append(sns.color_palette('gnuplot2', 7).as_hex()[5])
        elif v == 'β_pw':
            colors.append(sns.color_palette('gnuplot2', 7).as_hex()[6])
        elif v == 'β_w':
            colors.append('#ADFF2F')
        else:
            colors.append('000000')
    if len(colors) > 1:
        return colors
    else:
        return colors[0]


def load_beh(participant, block='', grating_freq=None):
    columns = ['direction', 'duration', 'trigger', 'stim_type', 'grating_freq',
               'corr_ans', 'resp.rt', 'resp.keys', 'resp.corr']
    if participant >200:
        p_beh = os.path.join('raw','beh', '{}_oddball_raw.csv'.format(participant))
    else:
        p_beh = os.path.join('raw','beh', '{}_oddball.csv'.format(participant))
    df = pd.read_csv(p_beh)
    df = df.replace(np.nan, 'None')
                    # keep_default_na=False, na_values=['NaN'])
    if block == 'decrement':
        df = df[df.direction == 'decreament']
    elif block == 'increment':
        df = df[df.direction == 'increament']
    else:
        df = df[:-1]

    if grating_freq is not None:
        df = df[df.grating_freq == grating_freq]
    df = df[columns]
    return df


def get_participants():
    p_list = list(range(201, 231))
    p_list.extend(list(range(232,242)))
    return p_list


def export_best_model():
    f_perf = os.path.join('ACT-R', 'Proj_TD','performance.csv')
    perf = pd.read_csv(f_perf)
    w = [0.5, 0.5]

    df = pd.DataFrame(columns=['participant', 'model', 'error_hr', 'error_cr'])
    p_list = np.unique(perf.participant).astype(int)
    models = ['nonadapt', 'adapt', 'novel']
    print('Summarizing fitting performance...', end='')
    for p in p_list:
        tp = perf[perf.participant==p]
        tp_0 = tp.loc[tp.type == 'human', ['HR', 'CR']].to_numpy()
        for m in models:
            tp_1 = tp.loc[tp.type == m, ['HR', 'CR']].to_numpy()
            err = (tp_1-tp_0)**2
            row = pd.DataFrame({'participant': p,
                                'model': m,
                                'error_hr': err[0,0],
                                'error_cr': err[0,1]}, index=[0])
            df = pd.concat([df, row], ignore_index=True)
    df['error'] = (df.error_hr*w[0]+df.error_cr*w[1])
    df = df.pivot(index='participant', columns='model', values='error')

    mat = np.array(df)
    best_fit = np.argmin(mat, axis=1)
    df['best'] = [df.columns[i] for i in best_fit]
    df = df.reset_index()
    print('Done.')
    df.to_csv('summary_performance.csv', index=False)
    return None


def plot_fitting_corr():
    import pingouin as pg
    f_perf = os.path.join('ACT-R', 'Proj_TD','performance.csv')
    df = pd.read_csv(f_perf)
    dmin, dmax = df['d-prime'].min(), df['d-prime'].max()
    groups = get_groups()
    f, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    for gi, (g, p_list) in enumerate(groups.items()):
        d_h = []
        d_m = []
        for p in p_list:
            r = np.logical_and(df.participant==p, df.type==g)
            d_m.append(df.loc[r, 'd-prime'].tolist()[0])
            r = np.logical_and(df.participant==p, df.type=='human')
            d_h.append(df.loc[r, 'd-prime'].tolist()[0])
        sns.regplot(x=d_m, y=d_h, ax=axes[gi], color=get_colors(g))
        axes[gi].set_xlim([dmin-0.1,dmax+0.1])
        axes[gi].set_ylim([dmin-0.1,dmax+0.1])
        if g == 'novel':
            axes[gi].set_title('oddball group')
        else:
            axes[gi].set_title('unadapted group')
        print(pg.linear_regression(d_m, d_h).round(3))
    axes[1].set_xlabel('model d\'')
    axes[0].set_ylabel('human d\'')
    axes[1].set_ylabel('human d\'')

    return None


def get_groups():
    f_load = 'summary_performance.csv'
    df = pd.read_csv(f_load)
    grouped = df.groupby(by=['best'])
    p_grouped = {}
    for g, idx in grouped.groups.items():
        p_grouped[g]=df.participant[idx].tolist()
    return p_grouped


def export_params_best_models():
    p_grouped = get_groups()
    params = pd.read_csv( os.path.join('ACT-R', 'Proj_TD','fitting_nonadapt.csv') ) # any
    par_cols = ['A0', 'a', 'start', 'A', 'B', 'n_step']
    for group, p_list in p_grouped.items():
        for p in p_list:
            params.loc[params.participant==p, 'best_model'] = group
            if group == 'nonadapt': continue
            tp = pd.read_csv( os.path.join('ACT-R', 'Proj_TD','fitting_{}.csv'.format(group)) )
            params.loc[params.participant==p, par_cols] = tp.loc[tp.participant==p, par_cols]
    params.to_csv('summary_model.csv', index=False)


def get_model_param(param, p, grating, direction):
    f_params = 'summary_model.csv'
    df = pd.read_csv(f_params)
    row = np.logical_and(df.participant==p, df.grating==grating)
    row = np.logical_and(row, df.direction==direction)
    return df.loc[row, param].tolist()[0]


def export_simulated_beh():
    models = pd.read_csv('summary_model.csv')
    participants = get_participants()
    for pi, p in enumerate(participants):
        m = models.loc[models.participant == p, 'best_model'].tolist()[0]
        sim_beh = pd.read_csv(os.path.join('ACT-R', 'Proj_TD', 'simulated_beh', '{}.csv'.format(p)))[m]
        sim_beh = sim_beh.replace(np.nan, 'None').tolist()

        beh = load_beh(p)
        beh['sim_resp'] = sim_beh
        beh['best_model'] = m
        beh.to_csv(os.path.join('results_sim', '{}.csv'.format(p)), index=False)
    return None


def export_d():
    subs = get_participants()
    conds = ['scaling', 'stim_type', 'grating_freq', 'direction']
    conds_vals = [[40,60,80,100,120,140,160], ['standard', 'oddball'],
                  [0,5,10,20], ['increament', 'decreament']]
    beh_indices = ['resp.corr']
    g = beh_group(subs, conds, conds_vals, beh_indices)
    g.plot_dprime(cond_x = 'scaling', x_vals = [40,60,80,120,140,160],
                  cond_group = 'grating_freq', group_vals = [0,5,10,20])
    data = g.d_group
    data.reset_index(inplace=True)
    data.drop(columns = 'index', inplace=True)
    data['direction'] = ''
    for i in range(data.shape[0]):
        if data.loc[i,'scaling'] < 100:
            data.loc[i, 'direction'] = 'decrement'
        else:
            data.loc[i, 'direction'] = 'increment'
    data.to_csv('summary_beh.csv', index=False)
    return None


def export_sim_d():
    from scipy.stats import norm
    participants = []
    scaling = []
    sim_d = []
    gratings = []
    direction = []
    for pi, p in enumerate(get_participants()):
        sim = pd.read_csv(os.path.join('results_sim', '{}.csv'.format(p)))
        sim = sim.replace(np.nan, 'None')
        for gi, g in enumerate([0,5,10,20]):
            tp = sim[sim.grating_freq==g]
            for ci, cond in enumerate([40,60,80,120,140,160]):
                if cond < 100:
                    tp2 = tp[tp.direction == 'decreament']
                    direc = 'decrement'
                else:
                    tp2 = tp[tp.direction == 'increament']
                    direc = 'increment'
                dn = tp2[tp2.duration == 0.6]
                dp = tp2[tp2.duration == 0.6*cond/100]
                n_pos = dp.shape[0]
                n_neg = dn.shape[0]
                hits = sum(dp.sim_resp == 'j')
                fas = sum(dn.sim_resp == 'j')
                # correct to avoid 0/1
                hr = (hits+0.1/3)/(n_pos+0.2/3)
                far = (fas+0.9)/(n_neg+1.8)
                d = norm.ppf(hr) - norm.ppf(far)
                participants.append(p)
                scaling.append(cond)
                sim_d.append(d)
                gratings.append(g)
                direction.append(direc)
    df = pd.DataFrame({'participant': participants,
                       'scaling': scaling,
                       'd': sim_d,
                       'grating_freq': gratings,
                       'direction': direction})
    df.to_csv('summary_sim.csv', index=False)


def examine_d_v2():  # with sim d
    import pingouin as pg
    sns.set_theme(style="white", font_scale=2)
    colors = sns.color_palette('flare_r', 4).as_hex()

    data = pd.read_csv('summary_beh.csv')
    data['change'] = data['scaling']-100

    sim = pd.read_csv('summary_sim.csv')
    data = data.merge(sim, on = ['participant', 'grating_freq', 'direction', 'scaling'], how='left')
    if False:
        data = data.melt(id_vars = ['participant', 'grating_freq', 'direction', 'scaling', 'change'], value_vars = ['d_x', 'd_y'],
                         var_name = 'dtype', value_name = 'd')
        data.loc[data['dtype'] == 'd_x', 'dtype'] = 'human'
        data.loc[data['dtype'] == 'd_y', 'dtype'] = 'model'

    f, axes = plt.subplots(nrows=1, ncols = 4, sharey=True, sharex=True)
    for gi, g in enumerate([0,5,10,20]):
        tp = data[data.grating_freq==g]
        sns.pointplot(data=tp, ax = axes[gi],
                      x='change', y='d_x',
                      color=colors[gi],
                      markers=["o"], linestyles=["-"],
                      errorbar='se')
        sns.pointplot(data=tp, ax = axes[gi],
                      x='change', y='d_y',
                      color=colors[gi],
                      markers=["s"], linestyles=["--"],
                      errorbar='se')
        if gi == 0:
            axes[gi].set(ylabel='d\'')
        else:
            axes[gi].set(ylabel = '')
        axes[gi].set(xlabel='Novelty (%)')
        axes[gi].set(title = '{} Hz'.format(g))

    f, axes = plt.subplots(nrows=1, ncols=2)
    sns.pointplot(data=data, ax = axes[0],
                  x='change', y = 'd_x', color = 'black',
                  markers = ['o'], linestyles=['-'],
                  errorbar='se')
    sns.pointplot(data=data, ax = axes[0],
                  x='change', y = 'd_y', color = 'black',
                  markers = ['s'], linestyles=['--'],
                  errorbar='se')
    axes[0].set(ylabel='d\'')
    axes[0].set(xlabel='Novelty (%)')

    sns.pointplot(data=data, ax = axes[1],
                  x='grating_freq', y='d_x',
                  palette='flare_r', markers = ['o'],
                  errorbar='se')
    axes[1].set(ylabel='d\'')
    axes[1].set(xlabel='Grating Frequency (Hz)')

    aov = pg.rm_anova(data=data,
                      dv='d_x',
                      within=['change', 'grating_freq'],
                      subject='participant',
                      effsize='np2').round(3)
    res = pg.pairwise_tests(data=data, dv='d_x', within='change', subject='participant').round(3)
    return aov, res

def calc_ticks(t0,a,duration=0.6):
    ticks = 0
    t = t0
    elapsed = 0
    err_min = 999
    err = 998
    while err<err_min:
        err_min = err
        elapsed += t
        err = (duration-elapsed)**2
        t *= a
        ticks += 1
    ticks -= 1
    return ticks

def get_clock_speed_nonadapt(p):
    # p - participant id
    beh = load_beh(p)
    beh.loc[beh.direction == 'decreament', 'direction'] = 'decrement'
    beh.loc[beh.direction == 'increament', 'direction'] = 'increment'
    beh['t0'] = 0
    beh['a'] = 0
    print('Adding a, t0 to dataframe...', end = '')
    for i in range(beh.shape[0]):
        beh.loc[i,'t0'] = get_model_param('start', p, beh['grating_freq'][i],  beh['direction'][i])
        beh.loc[i,'a'] = get_model_param('a', p, beh['grating_freq'][i], beh['direction'][i])
    print('Done.')
    return beh

def get_clock_speed_novel(p):
    beh = load_beh(p)
    beh.loc[beh.direction == 'decreament', 'direction'] = 'decrement'
    beh.loc[beh.direction == 'increament', 'direction'] = 'increment'
    beh['t0'] = 0
    beh['a'] = 0
    n_step = get_model_param('n_step', p, 5, 'increment')
    count0 = 0 # any trial
    count1 = 0 # consecutive no-responses
    print('Adding a, t0 to dataframe...', end='')
    for i in range(beh.shape[0]):
        count0 += 1
        count1 += 1
        if count0 == 1:
            m = beh.loc[i, 'grating_freq']
            d = beh.loc[i, 'direction']
            t0 = get_model_param('start', p, m, d)
            a = get_model_param('a', p, m, d)
        ans = beh.loc[i, 'resp.keys']
        if ans == 'None' and count1 == n_step and count0 <= 30:
            t0 += 0.001
            count1 = 0
        elif ans == 'j':
            t0 = get_model_param('start', p, m, d)
            count1 = 0
        if count0 == 30:
            count0 = 0
            count1 = 0
        beh.loc[i,'t0'] = t0
        beh.loc[i,'a'] = a
    print('Done.')
    return beh


def export_clock_speed(p):
    if get_model_param('best_model', p, 5, 'decrement') == 'nonadapt':
        df = get_clock_speed_nonadapt(p)
    else:
        df = get_clock_speed_novel(p)
    print('Adding average_ticks...', end='')
    df['lambda'] = 0
    for i in range(df.shape[0]):
        df.loc[i, 'lambda'] = calc_ticks(df.loc[i,'t0'], df.loc[i,'a'], df.loc[i,'duration'])
    print('Done.')
    df.to_csv(os.path.join('results_clock', '{}.csv'.format(p)), index=False)


def export_average_ticks():
    f_params = 'summary_model.csv'
    df = pd.read_csv(f_params)
    df['lambda'] = 0
    print('Updating...', end='')
    for i in range(df.shape[0]):
        p = df.loc[i, 'participant']
        m = df.loc[i, 'grating']
        d = df.loc[i, 'direction']
        beh = pd.read_csv(os.path.join('results_clock', '{}.csv'.format(p)))
        df.loc[i, 'lambda'] = beh.loc[np.logical_and(beh.grating_freq==m, beh.direction==d), 'lambda'].to_numpy().mean()
    df.to_csv(f_params, index=False)
    print('Done.')


def update_clock_beh():
    f_params = 'summary_model.csv'
    df = pd.read_csv(f_params)
    df['d-prime'] = 0
    beh = pd.read_csv('summary_beh.csv')
    print('Updating...', end='')
    for i in range(df.shape[0]):
        p = df.loc[i, 'participant']
        m = df.loc[i, 'grating']
        d = df.loc[i, 'direction']
        idx = np.logical_and(beh.participant == p, beh.grating_freq == m)
        idx = np.logical_and(idx, beh.direction == d)
        df.loc[i, 'd-prime'] = beh.loc[idx, 'd'].to_numpy().mean()
    df.to_csv(f_params, index=False)
    print('Done.')


def update_clock_600():
    f_params = 'summary_model.csv'
    df = pd.read_csv(f_params)
    df['lambda_600'] = 0
    print('Updating...', end='')
    for i in range(df.shape[0]):
        p = df.loc[i, 'participant']
        m = df.loc[i, 'grating']
        d = df.loc[i, 'direction']
        beh = pd.read_csv(os.path.join('results_clock', '{}.csv'.format(p)))
        beh = beh[beh.stim_type == 'standard']
        df.loc[i, 'lambda_600'] = beh.loc[np.logical_and(beh.grating_freq==m, beh.direction==d), 'lambda'].to_numpy().mean()
    df.to_csv(f_params, index=False)
    print('Done.')



def examine_beh_v3():  # with outliers removed
    df = pd.read_csv('summary_model.csv')
    dat = df[['A', 'lambda', 'd-prime']]
    dat = remove_outliers_mahalanobis(dat.to_numpy(dtype='float32'))
    dat = pd.DataFrame(dat, columns = ['A', 'lambda', 'd-prime'])
    lm = pg.linear_regression(X=dat[['A', 'lambda']], y=dat['d-prime'])
    print(lm.round(3))
    f, axes = plt.subplots(nrows=2, ncols=1, sharey=True)
    sns.regplot(ax=axes[0], data = dat, x='A', y='d-prime')
    sns.regplot(ax=axes[1], data = dat, x='lambda', y='d-prime')
    axes[0].set_ylabel('d\'')
    axes[1].set_ylabel('d\'')
    axes[0].set_xlabel('Decision Threshold (A)')
    axes[1].set_xlabel('Clock Speed (λ)')


def load_erp_decrement(p, channel):
    channels = load_channels()
    participants = get_participants()
    ci = np.where( np.array(channels) == channel )[0][0]
    pi = np.where( np.array(participants) == p )[0][0]
    files = ['erp_offset_standard_de.mat',
             'erp_offset_minus_20.mat',
             'erp_offset_minus_40.mat',
             'erp_offset_minus_60.mat']
    dat = []
    for f in files:
        tp = loadmat(os.path.join('results_ERP', f))['erp_all']
        dat.append(tp[ci,:,pi])
    timeline = loadmat(os.path.join('results_ERP', f))['times']
    return np.array(dat), timeline.flatten()


def load_erp_increment(p, channel):
    channels = load_channels()
    participants = get_participants()
    ci = np.where( np.array(channels) == channel )[0][0]
    pi = np.where( np.array(participants) == p )[0][0]
    files = ['erp_offset_standard_in.mat',
             'erp_offset_plus_20.mat',
             'erp_offset_plus_40.mat',
             'erp_offset_plus_60.mat']
    dat = []
    for f in files:
        tp = loadmat(os.path.join('results_ERP', f))['erp_all']
        dat.append(tp[ci,:,pi])
    timeline = loadmat(os.path.join('results_ERP', f))['times']
    return np.array(dat), timeline.flatten()


def plot_novelty_effect(direction, ax=None):
    baseline = [-50,50]
    if direction == 'decrement':
        _, timeline = load_erp_decrement(201, 'Pz')
    else:
        _, timeline = load_erp_increment(201, 'Pz')
    id0 = np.argmin(np.abs(timeline - baseline[0]))
    id1 = np.argmin(np.abs(timeline - baseline[1]))
    erp = []
    p_list = get_participants()
    print('Loading data...', end='')
    for pi, p in enumerate(p_list):
        if direction == 'decrement':
            tp, timeline = load_erp_decrement(p, 'Pz')
        else:
            tp, timeline = load_erp_increment(p, 'Pz')
        tp = tp - np.mean(tp[:, id0:id1+1], axis=1, keepdims=True)
        erp.append(tp)
    erp_n = np.array(erp)
    #erp = np.nanmean(erp_n, axis=0)
    print('Done.')

    for i in range(erp_n.shape[1]):
        for pi, p in enumerate(p_list):
            tp = pd.DataFrame({'change': '{}%'.format(i*20),
                               'participant': p,
                               'erp': erp_n[pi,i,:],
                               'time': timeline})
            if i == 0 and pi == 0:
                df = tp.copy()
            else:
                df = pd.concat([df, tp], ignore_index=True)
    if ax is None:
        f = plt.figure()
        sns.lineplot(data=df, x='time', y='erp', hue='change', palette='cool', linewidth=3, errorbar ='se')
    else:
        sns.lineplot(ax=ax, data=df, x='time', y='erp', hue='change', palette='cool', linewidth=3, errorbar ='se')

    return erp_n


def get_amplitude(erp, timeline, timewin):
    # erp (n, c, t)
    import pingouin as pg
    id0 = np.argmin(np.abs(timeline - timewin[0]))
    id1 = np.argmin(np.abs(timeline - timewin[1]))
    amp = np.mean(erp[:,:,id0:id1+1], axis=2)
    for i in range(amp.shape[1]):
        tp = pd.DataFrame({'change_int': i*20,
                           'change': '{}%'.format(20*i),
                           'pid': range(amp.shape[0]),
                           'amp': amp[:,i]})
        if i == 0:
            df = tp.copy()
        else:
            df = pd.concat([df, tp], ignore_index=True)
    return df


def get_novelty_effect(p_list, direction):
    if direction == 'decrement':
        win = [400, 600]
        _, timeline = load_erp_decrement(201, 'Pz')
    else:
        win = [200, 400]
        _, timeline = load_erp_increment(201, 'Pz')
    id0 = np.argmin(np.abs(timeline - win[0]))
    id1 = np.argmin(np.abs(timeline - win[1]))
    amp = []
    print('Loading data...', end='')
    for p in p_list:
        if direction == 'decrement':
            tp, timeline = load_erp_decrement(p, 'Pz')
        else:
            tp, timeline = load_erp_increment(p, 'Pz')
        tp = tp[:, id0:id1+1]
        tp2 = []
        for i in range(1,tp.shape[0]):
            tp2.append(tp[i,:]-tp[0,:])
        amp.append(np.array(tp2).mean(axis=1))
    print('Done.')
    amp = pd.DataFrame(np.array(amp), columns = ['20%', '40%', '60%'])
    amp['participant'] = p_list
    amp = pd.melt(amp, id_vars=['participant'], value_vars=['20%', '40%', '60%'],
                  var_name = 'change', value_name = 'amp')

    return amp


def plot_novelty_erp_stats():
    import pingouin as pg
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax0 = plt.subplot2grid(shape=(2,4), loc=(0,0), colspan=2)
    ax1 = plt.subplot2grid(shape=(2,4), loc=(1,0), colspan=2)
    ax2 = plt.subplot2grid(shape=(2,4), loc=(0,2))
    ax3 = plt.subplot2grid(shape=(2,4), loc=(1,2))
    ax4 = plt.subplot2grid(shape=(2,4), loc=(0,3))
    ax5 = plt.subplot2grid(shape=(2,4), loc=(1,3))
    erp_de = plot_novelty_effect('decrement', ax0)
    erp_in = plot_novelty_effect('increment', ax1)
    ax0.set_xlabel('')
    ax0.set_ylabel('ERP (μV)')
    ax0.legend(loc="upper left", ncol=2)
    ax0.set_xlim([-100, 600])
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('ERP (μV)')
    ax1.legend().remove()
    ax1.set_xlim([-100, 600])

    _, timeline = load_erp_decrement(201, 'Pz')
    amp_de = get_amplitude(erp_de, timeline, [400, 600])
    amp_in = get_amplitude(erp_in, timeline, [200, 400])
    amp_de.dropna(inplace=True)
    amp_in.dropna(inplace=True)
    sns.regplot(ax=ax2, data=amp_de, x='change_int', y='amp')
    sns.regplot(ax=ax3, data=amp_in, x='change_int', y='amp')
    #sns.violinplot(ax=ax2, data=amp_de, x='change', y='amp', hue='change', palette='cool')
    #sns.violinplot(ax=ax3, data=amp_in, x='change', y='amp', hue='change', palette='cool')
    ax2.set_xlabel('')
    ax2.set_ylabel('Amplitude (μV)')
    ax2.set_xticks([0,20,40,60])
    ax2.set_xticklabels(['0%', '20%', '40%', '60%'])
    ax2.legend().remove()
    ax3.set_xlabel('')
    ax3.set_ylabel('Amplitude (μV)')
    ax3.set_xticks([0, 20, 40, 60])
    ax3.set_xticklabels(['0%', '20%', '40%', '60%'])
    ax3.legend().remove()
    print(pg.linear_regression(X=amp_de.change_int, y=amp_de.amp).round(3))
    print(pg.linear_regression(X=amp_in.change_int, y=amp_in.amp).round(3))

    groups = get_groups()
    for direction in ['decrement', 'increment']:
        for gi, (g, p_list) in enumerate(groups.items()):
            tp = get_novelty_effect(p_list, direction)
            tp['group'] = g
            if gi == 0:
                df = tp.copy()
            else:
                df = pd.concat([df, tp], ignore_index=True)
        if direction == 'decrement':
            diff_de = df.copy()
        else:
            diff_in = df.copy()

    diff_de.dropna(inplace=True)
    diff_in.dropna(inplace=True)
    sns.violinplot(ax=ax4, data=diff_de, x='group', y='amp', hue='group', palette=get_colors(['nonadapt', 'novel']))
    sns.violinplot(ax=ax5, data=diff_in, x='group', y='amp', hue='group', palette=get_colors(['nonadapt', 'novel']))
    ax4.set_xlabel('')
    ax4.set_xticklabels(['unadapted', 'oddball'])
    ax4.set_ylabel('Novelty effect (μV)')
    ax4.legend().remove()
    ax5.set_xlabel('')
    ax5.set_xticklabels(['unadapted', 'oddball'])
    ax5.set_ylabel('Novelty effect (μV)')
    ax5.legend().remove()
    print(pg.ttest(diff_de.loc[diff_de.group=='nonadapt','amp'], diff_de.loc[diff_de.group=='novel','amp']).round(3))
    print(pg.ttest(diff_in.loc[diff_de.group=='nonadapt','amp'], diff_in.loc[diff_de.group=='novel','amp']).round(3))

def load_task_psd(subject, grating_freq):
    f_load = os.path.join('data_psd', '{}Hz'.format(grating_freq), '{}.mat'.format(subject))
    dat  = loadmat(f_load)
    return dat['psd_rs'], dat['psd_st'], dat['frex'][0], dat['srate'][0][0]

def convert_to_fooof_features(fm): # fm - fooof object, fitted
    labels = ['exponent', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    bands = [[0, 4], [4, 7.5], [7.5, 13], [13, 30], [30, 40]]
    feats = [fm.aperiodic_params_.tolist()]
    if len(feats[0]) == 2:
        feats[0].append(0)
    for band in bands:
        pidx = np.logical_and(fm.peak_params_[:,0] > band[0], fm.peak_params_[:,0] <= band[1])
        peak = fm.peak_params_[pidx,:]
        if peak.shape[0] == 0:
            peak = [0,0,0]
        else:
            peak = peak[np.argmax(peak[:,1]),:].tolist()
        feats.append(peak)
    return np.array(feats), labels, fm.r_squared_

def export_task_fooof_features(p, grating_freq): # p - participant
    from fooof import FOOOF
    print('Exporting fooof features participant = {}, motion = {}Hz ...'.format(p, grating_freq), end = '')
    [psd_rs, psd_st, frex, srate] = load_task_psd(p, grating_freq)
    feats_rs = []
    feats_st = []
    for chani in range(psd_rs.shape[0]):
        for trial in range(psd_rs.shape[2]):
            fm = FOOOF(peak_width_limits=[4,12])
            fm.fit(frex, psd_rs[chani,:,trial], [0.5,40])
            if False: fm.plot()
            [feats, labels, r2] = convert_to_fooof_features(fm)
            tp = [chani, trial]
            tp.extend(feats.flatten().tolist())
            tp.append(r2)
            feats_rs.append(tp)

    for chani in range(psd_st.shape[0]):
        for trial in range(psd_st.shape[2]):
            fm = FOOOF(peak_width_limits=[4,12])
            fm.fit(frex, psd_st[chani,:,trial], [0.5,40])
            if False: fm.plot()
            [feats, labels, r2] = convert_to_fooof_features(fm)
            tp = [chani, trial]
            tp.extend(feats.flatten().tolist())
            tp.append(r2)
            feats_st.append(tp)
    columns = ['channel', 'trial']
    labels = np.array(list(product(labels, [0,1,2])))
    labels = np.char.add(labels[:,0], labels[:,1]).tolist()
    columns.extend(labels)
    columns.append('R2')
    feats_rs = pd.DataFrame(np.array(feats_rs), columns=columns)
    feats_st = pd.DataFrame(np.array(feats_st), columns=columns)

    f_save = os.path.join('data_fooof', '{}Hz'.format(grating_freq))
    feats_rs.to_csv(os.path.join(f_save, 'rs', '{}.csv'.format(p)), index=False)
    feats_st.to_csv(os.path.join(f_save, 'st', '{}.csv'.format(p)), index=False)
    print('Done.')
    return None


def export_fooof_batch():
    for freq in [0,5,10,20]:
        for p in get_participants():
            export_task_fooof_features(p, freq)
    return None

def get_task_fooof(p, grating_freq, cond, r2_cutoff=None): # p-participant, cond-['rs', 'st']
    f_save = os.path.join('data_fooof', '{}Hz'.format(grating_freq))
    feats = pd.read_csv(os.path.join(f_save, cond, '{}.csv'.format(p)))
    if r2_cutoff is None:
        return feats
    else:
        return feats[feats.R2 > r2_cutoff]

def flatten_features(df): # df - (channel, trial, ..., R2)
    trials = np.unique(df.trial)
    feats = []
    for trial in trials:
        df[df.trial==trial].iloc[:,2:20]
        feats.append(df[df.trial==trial].iloc[:,2:20].to_numpy().flatten())
    feats = np.array(feats)
    tp_l = df.columns[2:20].to_numpy()
    tp_l = np.array(list(product(range(int(feats.shape[1]/18)), tp_l)))
    tp_l0 = np.array(['chan_']*tp_l.shape[0])
    tp_l1 = np.array(['_'] * tp_l.shape[0])
    labels = np.char.add(tp_l0, tp_l[:,0])
    labels = np.char.add(labels, tp_l1)
    labels = np.char.add(labels, tp_l[:,1]).tolist()
    return feats, labels

def get_fooof_features(cond, participants = get_participants(), motions = [0,5,10,20],
                       feats_use=None, include_nan=False, verbose=True):
    # cond = 'rs', 'st'
    if type(participants) == int:
        participants = [participants]
    if type(motions) == int:
        motions = [motions]
    if verbose: print('Loading data...', end = '')
    for pi, p in enumerate(participants):
        for fi, freq in enumerate(motions):
            tp = get_task_fooof(p, grating_freq=freq, cond=cond, r2_cutoff=None)
            [tp, labels] = flatten_features(tp)
            if pi == 0 and fi == 0:
                feats = tp.copy()
            else:
                feats = np.concatenate([feats, tp.copy()], axis=0)
    if verbose: print('Done.')
    if feats_use is not None:
        labels = np.array(labels)
        feats = feats[:, np.isin(labels, feats_use)]
        labels = labels[np.isin(labels, feats_use)].tolist()
    if include_nan:
        feats[feats==0] = np.nan
    return feats, labels

def class_size(y, normOn=False): # count the number of each category
    yvals = np.unique(y)
    y = np.array(y)
    res = {}
    for yval in yvals:
        if not normOn:
            res[yval] = np.sum(y == yval)
        else:
            res[yval] = np.sum(y == yval)/ y.shape[0]
    return res

def cv_splits(n_cv, X,y, shuffle=True):
    # X(None, dim0, dim1)
    # each split contains balanced samples
    yvals = np.unique(y)
    y = np.array(y)
    y_idx = {}
    for yi, yval in enumerate(yvals):
        y_idx[yval] = np.where(y==yval)[0]
        if shuffle:
            np.random.shuffle(y_idx[yval])

    y_class  = class_size(y, normOn=True)
    split_size = np.ceil(X.shape[0] / n_cv).astype(int)  # all yvals in each split
    idx_split =[]
    for cvi in range(n_cv):
        idx_split.append([])
        for yval, yval_idx in y_idx.items():
            size0 = np.round(split_size * y_class[yval]).astype(int)
            idx_split[cvi].extend(yval_idx[size0*cvi:size0*(cvi+1)])
    # class_size(y[idx_split[0]])

    idx_test = idx_split.copy()
    idx_train = []
    for cvi0 in range(n_cv):
        tp = np.array([])
        for cvi1 in range(n_cv):
            if cvi1 == cvi0:
                continue
            else:
                tp = np.concatenate([tp, idx_split[cvi1]])
        idx_train.append(tp.astype(int).tolist())
    return idx_train, idx_test


def get_metric(y_hat, y):
    y_hat = np.array(y_hat)
    y = np.array(y)
    yvals = np.unique(y)
    tr = {}
    for yi, yval in enumerate(yvals):
        tr[yval] = np.sum(y[y==yval] == y_hat[y==yval]) / np.sum(y==yval)
    acc = np.sum(y_hat == y)/ y.shape[0]
    return acc, tr


def get_task_data(p, motion=[0,5,10,20], feats_use=None, verbose=True, include_nan=False): # X,y to model
    conds = ['rs', 'st']
    for ci, cond in enumerate(conds):
        [tp_x, feature_names] = get_fooof_features(cond=cond, participants=p, motions=motion,
                                                   feats_use=feats_use, include_nan=include_nan, verbose=verbose)
        tp_y = [ci]*tp_x.shape[0]
        if ci == 0:
            X = tp_x.copy()
            Y = tp_y.copy()
        else:
            X = np.concatenate([X, tp_x])
            Y.extend(tp_y)
    Y = np.array(Y)
    return X, Y


def training_svm(p, batch, run = None, final_model = False, test_features = True, feats_use = None,
                 shuffle_y = False):
    if not shuffle_y:
        if not os.path.exists('results_svm'): os.mkdir('results_svm')
        f_main = os.path.join('results_svm', 'svm_batch_{}'.format(batch))
    else:
        if not os.path.exists(os.path.join('results_svm', 'permutation')):
            os.makedirs(os.path.join('results_svm', 'permutation'))
        f_main = os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch))
    if not os.path.exists(f_main): os.makedirs(f_main)

    if not final_model and run is None:
        print('Error: run has to be specified for cross-validation.')
        return
    elif not final_model and run is not None:
        if not os.path.exists(os.path.join(f_main, 'feature_selection_run_{}'.format(run))):
            os.makedirs(os.path.join(f_main, 'feature_selection_run_{}'.format(run)))
    else: # final_model
        if not os.path.exists(os.path.join(f_main, 'final_models')):
            os.makedirs(os.path.join(f_main, 'final_models'))

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVC
    warnings.filterwarnings('ignore')

    [X,y] = get_task_data(p, feats_use=feats_use, include_nan=True)
    if shuffle_y:
        np.random.shuffle(y)

    if not final_model:
        n_split = 5
        [idx_train, idx_test] = cv_splits(n_split, X, y, shuffle=True)
    else:
        n_split = 1
        idx_train = [list(range(X.shape[0]))]
        idx_test = idx_train.copy()

    acc_train = []
    acc_test = []
    sen_train = []
    spe_train = []
    sen_test = []
    spe_test = []
    C_list = [1e-5,1e-4,1e-3,1e-2, 1e-1, 1, 10,100]

    for i in range(0,n_split):
        t0 = time()
        X_train = X[idx_train[i],:]
        X_test  = X[idx_test[i],:]
        y_train = y[idx_train[i]]
        y_test  = y[idx_test[i]]
        scores  = []
        print('Fitting SVM...{}/{}...'.format(i+1, n_split), end='')
        for C in C_list:
            clf = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                                StandardScaler(),
                                SVC(kernel='rbf', C=C, gamma='scale', probability=final_model))
            clf.fit(X_train,y_train)
            scores.append(clf.score(X_test, y_test))
        C = C_list[np.argmax(scores)]
        clf = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                            StandardScaler(),
                            SVC(kernel='rbf', C=C, gamma='scale', probability=final_model))
        clf.fit(X_train, y_train)
        t1 = time()
        print('Done. Time cost: {:3.2f} min'.format( (t1-t0)/60 ))

        if not final_model:
            f_save = os.path.join(f_main, 'feature_selection_run_{}'.format(run), 'p_{}_cv_{}.pkl'.format(p,i))
        else:
            f_save = os.path.join(f_main, 'final_models', 'p_{}_cv_{}.pkl'.format(p,99))

        with open( f_save, 'wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

        if False:
            with open(f_save, 'rb') as f:
                clf = pickle.load(f)
        print('Predicting...', end ='')
        t0 = time()
        #pr_y_train = clf.predict_proba(X_train)
        yhat_train = np.array(clf.predict(X_train))
        yhat_test = np.array(clf.predict(X_test))
        [tpa_train, tr_train] = get_metric(yhat_train, y_train)
        [tpa_test, tr_test] = get_metric(yhat_test, y_test)
        acc_train.append(tpa_train)
        acc_test.append(tpa_test)
        sen_train.append(tr_train[1])
        sen_test.append(tr_test[1])
        spe_train.append(tr_train[0])
        spe_test.append(tr_test[0])
        t1 = time()
        print('Done. Time cost: {:3.2f} min'.format( (t1-t0)/60 ))

        if test_features:
            print('Testing features...')
            t0 = time()
            if i == 0:
                acc_feat = []
            tp_acc = []
            for j in range(X.shape[1]):
                tp_test = X_test.copy()
                tp_test[:,j] = np.nan # mute the feature
                tp_acc.append( clf.score(tp_test, y_test) )
                print('{:3.2f}%'.format(j/X.shape[1]*100), end='\r')
            acc_feat.append(np.array(tp_acc)/tpa_test) # accuracy change
            t1 = time()
            print('Done. Time cost: {:3.2f} min'.format((t1 - t0) / 60))

    if test_features:
        acc_feat = np.array(acc_feat)
        [_, feature_names] = get_fooof_features(participants=p, cond='rs', feats_use=feats_use)
        f_save = os.path.join(f_main, 'feature_selection_run_{}'.format(run), 'p_{}_test_features.csv'.format(p))
        pd.DataFrame(acc_feat, columns=feature_names).to_csv(f_save, index=False)
    arr = np.array([acc_train, acc_test, sen_train, sen_test, spe_train, spe_test]).transpose([1,0])
    df = pd.DataFrame(arr, columns = ['acc_train', 'acc_test', 'sen_train', 'sen_test', 'spe_train', 'spe_test'])

    if not final_model:
        df.to_csv(os.path.join(f_main, 'feature_selection_run_{}'.format(run), 'p_{}_performance.csv'.format(p)))
    else:
        df.to_csv(os.path.join(f_main, 'final_models', 'p_{}_performance_final.csv'.format(p)))
    return df


def eval_perf(run, batch, perm_batch = False):
    if not perm_batch:
        f_main = os.path.join('results_svm', 'svm_batch_{}'.format(batch))
    else:
        f_main = os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch))
    perf = []
    for pi, p in enumerate(get_participants()):
        tp = pd.read_csv(os.path.join(f_main, 'feature_selection_run_{}'.format(run),
                                      'p_{}_performance.csv'.format(p)))
        perf.append(np.mean(tp.iloc[:,1:],axis=0).tolist())

    perf = np.array(perf)
    df = pd.DataFrame(perf, columns=tp.columns[1:], index = get_participants())
    feats = pd.read_csv(os.path.join(f_main, 'feature_selection_run_{}'.format(run),
                                     'p_201_test_features.csv'))
    return df, feats.shape[1]


def eval_features(run, batch, perm_batch = False):
    if not perm_batch:
        f_main = os.path.join('results_svm', 'svm_batch_{}'.format(batch))
    else:
        f_main = os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch))
    warnings.filterwarnings('ignore')
    for pi, p in enumerate(get_participants()):
        tp = pd.read_csv(os.path.join(f_main, 'feature_selection_run_{}'.format(run),
                                      'p_{}_test_features.csv'.format(p)))
        if pi == 0:
            df = tp.copy()
        else:
            df = pd.concat([df, tp])
    pvals =[]
    for i in range(df.shape[1]):
        pvals.append(pg.ttest(df.iloc[:, i], 1)['p-val'].tolist()[0])
    idx = np.logical_and(np.array(pvals) < 0.05, df.mean() < 1)
    features = df.columns[idx].tolist()
    return features


def plot_perf_over_runs(max_run, batch, perm_batch=False):
    nfeats = []
    for run in range(1,max_run+1):
        perf, n = eval_perf(run, batch, perm_batch)
        perf['bacc_train'] = (perf['spe_train'] + perf['sen_train'])/2
        perf['bacc_test'] = (perf['spe_test'] + perf['sen_test'])/2
        perf['nfeat'] = n
        perf['run'] = run
        perf['participant'] = perf.index
        if run == 1:
            df = perf.copy()
        else:
            df = pd.concat([df, perf], ignore_index=False)
        nfeats.append(n)
    df = pd.melt(df, id_vars=['participant', 'run', 'nfeat'], value_vars=['bacc_train', 'bacc_test'],
                 var_name = 'type', value_name = 'bacc')
    f = plt.figure()
    sns.pointplot(data = df, x = 'run', y ='bacc', hue='type', palette='BuPu_r')
    plt.legend().remove()
    plt.xlabel('#feature')
    plt.xticks(ticks = range(max_run), labels = nfeats)
    #plt.xlim([540,0])


def get_perf_over_batch():
    batches = range(1,101)
    print('Getting data...')
    bacc_train = []
    bacc_test = []
    nfeat = []
    max_run_list = []
    batch_list = []
    for bi, batch in enumerate(batches):
        f_main = os.path.join('results_svm', 'svm_batch_{}'.format(batch))
        files = os.listdir(f_main)
        run_files = [f for f in files if '_run_' in f]
        runs = [int(f.split('_')[-1]) for f in run_files]
        max_run = max(runs)  # final run

        perf, n = eval_perf(max_run, batch, perm_batch=False)
        bacc_train.append(np.mean((perf['spe_train'] + perf['sen_train'])/2))
        bacc_test.append(np.mean((perf['spe_test'] + perf['sen_test'])/2))
        nfeat.append(n)
        max_run_list.append(max_run)
        batch_list.append(batch)
        print('{:3.2f}%'.format(bi/len(batches)*100), end = '\r')
    print('Done.')
    df = pd.DataFrame({'bacc_train': bacc_train,
                       'bacc_test': bacc_test,
                       'nfeat': nfeat,
                       'max_run': max_run_list,
                       'batch': batch_list})

    print(df.describe().round(3))


def svm_batch(batch, shuffle_y = False):
    t0 = time()
    n_feat = []

    # cross-validation (run=1)
    run = 1
    n_feat.append(540)
    for p in get_participants():
        training_svm(p, final_model=False, feats_use=None, test_features=True,
                     run=run, batch=batch, shuffle_y=shuffle_y)

    # cross-validation (run>1)
    run = 2
    feats_use = eval_features(run=run-1, batch=batch, perm_batch=shuffle_y) # use the previous run
    n_feat.append(len(feats_use))
    while n_feat[-1] < n_feat[-2]:
        for p in get_participants():
            training_svm(p, final_model=False, feats_use=feats_use, test_features=True,
                         run=run, batch=batch, shuffle_y=shuffle_y)
        run += 1
        feats_use = eval_features(run=run-1, batch=batch, perm_batch=shuffle_y)  # use the previous run
        n_feat.append(len(feats_use))
    print('Feature number converged at {}'.format(n_feat[-1]))

    if False:
        # final model (not used in the following analysis)
        run = len(n_feat) - 1  # final run
        feats_use = eval_features(run=run) # use the final run
        for p in get_participants():
            training_svm(p, batch = batch, final_model=True, feats_use=feats_use,
                         test_features=False)
    print('Total time for the batch: {:3.2} hrs'.format( (time()-t0) / 3600 ))


if False:
    perm_batch = False  # False or True for real and perm data respectively
    for batch in range(1,101):
        if not perm_batch:
            f_main = os.path.join('results_svm', 'svm_batch_{}'.format(batch))
        else:
            f_main = os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch))
        files = os.listdir(f_main)
        run_files = [f for f in files if '_run_' in f]
        runs = [int(f.split('_')[-1]) for f in run_files]
        run = max(runs)  # final run

        #plot_perf_over_runs(run, batch, perm_batch)
        export_percept_features_batch(run, batch, perm_batch)
        export_stat_percept_features(batch, perm_batch)
        print('Batch {} is done.'.format(batch))


def interpret_feat(feat, task_feat=True):
    feat = feat.split('_')
    if feat[2][:3] == 'exp':
        if feat[2][-1] == '0': tp = 'AP_o'
        if feat[2][-1] == '1': tp = 'AP_e'
        if feat[2][-1] == '2': tp = 'AP_k'
    elif feat[2][:3] == 'alp':
        if feat[2][-1] == '0': tp = 'α_f'
        if feat[2][-1] == '1': tp = 'α_pw'
        if feat[2][-1] == '2': tp = 'α_w'
    elif feat[2][:3] == 'bet':
        if feat[2][-1] == '0': tp = 'β_f'
        if feat[2][-1] == '1': tp = 'β_pw'
        if feat[2][-1] == '2': tp = 'β_w'
    elif feat[2][:3] == 'del':
        if feat[2][-1] == '0': tp = 'δ_f'
        if feat[2][-1] == '1': tp = 'δ_pw'
        if feat[2][-1] == '2': tp = 'δ_w'
    elif feat[2][:3] == 'gam':
        if feat[2][-1] == '0': tp = 'γ_f'
        if feat[2][-1] == '1': tp = 'γ_pw'
        if feat[2][-1] == '2': tp = 'γ_w'
    elif feat[2][:3] == 'the':
        if feat[2][-1] == '0': tp = 'θ_f'
        if feat[2][-1] == '1': tp = 'θ_pw'
        if feat[2][-1] == '2': tp = 'θ_w'
    return get_channel(int(feat[1]), task_feat), tp


def interpret_feats(feats, task_feat=True):
    res = []
    for feat in feats:
        res.append(interpret_feat(feat, task_feat))
    return np.array(res)


def export_percept_features_batch(run, batch, perm_batch = False):
    participants = get_participants()
    motions = [0,5,10,20]
    feats = eval_features(run, batch, perm_batch)
    dat = []
    counti = 0
    print('Getting data...')
    for feat in feats:
        for p in participants:
            [X1, flabels] = get_fooof_features(cond='st', participants=p, motions=motions, feats_use=[feat], verbose=False)
            [X0, flabels] = get_fooof_features(cond='rs', participants=p, motions=motions, feats_use=[feat], verbose=False)
            dat.append([feat, p, np.mean(X1-X0)])
            counti += 1
            print('{:3.2f}%'.format(counti/len(feats)/len(participants)*100), end='\r')
    print('Done.')
    df = pd.DataFrame(np.array(dat), columns = ['feat', 'participant', 'on-off'])
    print('Interpreting the features...', end = '')
    df['channel'] = interpret_feats(df.feat)[:,0]
    df['feature'] = interpret_feats(df.feat)[:,1]
    print('Done.')
    if not perm_batch:
        df.to_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features.csv'), index=False)
    else:
        df.to_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features.csv'), index=False)
    return None


def export_percept_features(): # using features with likelihood above all perm likelihood
    participants = get_participants()
    motions = [0,5,10,20]
    df = plot_likelihood_distribution(True)
    feats = df.feat.tolist()
    dat = []
    counti = 0
    print('Getting data...')
    for feat in feats:
        for p in participants:
            [X1, flabels] = get_fooof_features(cond='st', participants=p, motions=motions, feats_use=[feat], verbose=False)
            [X0, flabels] = get_fooof_features(cond='rs', participants=p, motions=motions, feats_use=[feat], verbose=False)
            dat.append([feat, p, np.mean(X1-X0)])
            counti += 1
            print('{:3.2f}%'.format(counti/len(feats)/len(participants)*100), end='\r')
    print('Done.')
    df = pd.DataFrame(np.array(dat), columns = ['feat', 'participant', 'on-off'])
    print('Interpreting the features...', end = '')
    df['channel'] = interpret_feats(df.feat)[:,0]
    df['feature'] = interpret_feats(df.feat)[:,1]
    print('Done.')
    df.to_csv('percept_features.csv', index=False)
    return None


def export_stat_percept_features(batch = None, perm_batch = False):
    if batch is None:
        df = pd.read_csv('percept_features.csv')
    else:
        if not perm_batch:
            df = pd.read_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features.csv'))
        else:
            df = pd.read_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features.csv'))

    feats = np.unique(df.feat)
    import pingouin as pg
    print('Getting stats...', end = '')
    for fi, feat in enumerate(feats):
        tp = pg.ttest(df.loc[df.feat==feat, 'on-off'], y=0)
        tp.index = [feat]
        if fi == 0:
            res = tp.copy()
        else:
            res = pd.concat([res,tp], axis=0)
    print('Done.')
    res.reset_index(inplace=True)
    if batch is None:
        res.to_csv('percept_features_stats.csv', index=False)
    else:
        if not perm_batch:
            res.to_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'), index=False)
        else:
            res.to_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'), index=False)
    return None


def plot_percept_features(batch=None, perm_batch = False):
    if batch is None:
        df = pd.read_csv('percept_features.csv')
        stats = pd.read_csv('percept_features_stats.csv')
    else:
        if not perm_batch:
            df = pd.read_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features.csv'))
            stats = pd.read_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'))
        else:
            df = pd.read_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features.csv'))
            stats = pd.read_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'))
    stats = stats[stats['p-val']<0.05]
    df = df[np.isin(df.feat, stats['index'])]
    df.sort_values(by='feature', inplace=True)

    feats_plot = np.unique(df.feature)
    colors = get_colors(feats_plot)
    sns.set_theme(style="white", font_scale=2)
    f, axes = plt.subplots(nrows = 1, ncols = 5)
    for fi, feat in enumerate(feats_plot):
        tp = df[df.feature==feat]
        sns.barplot(ax = axes[fi], data=tp, x='channel', y='on-off',
                    color=colors[fi], width=0.002*tp.shape[0])
        axes[fi].set_title(feat)
        axes[fi].set_xlabel('')
        if fi == 0:
            axes[fi].set_ylabel('( stimulus on - stimulus off )')
        else:
            axes[fi].set_ylabel('')
        axes[fi].set_xticklabels(axes[fi].get_xticklabels(), rotation=90)
    return None


def plot_percept_features_batches(batches, perm_batch = False):
    sns.set_theme(style="white", font_scale=2)
    f, axes = plt.subplots(nrows=len(batches), ncols=7)
    for bi, batch in enumerate(batches):
        if not perm_batch:
            df = pd.read_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features.csv'))
            stats = pd.read_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'))
        else:
            df = pd.read_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features.csv'))
            stats = pd.read_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'))
        stats = stats[stats['p-val'] < 0.05]
        df = df[np.isin(df.feat, stats['index'])]
        df.sort_values(by='feature', inplace=True)
        feats_plot = np.unique(df.feature)
        colors = get_colors(feats_plot)
        for fi, feat in enumerate(feats_plot):
            tp = df[df.feature == feat]
            sns.barplot(ax=axes[bi, fi], data=tp, x='channel', y='on-off',
                        color=colors[fi], width=0.002 * tp.shape[0])
            axes[bi, fi].set_title(feat)
            axes[bi, fi].set_xlabel('')
            axes[bi, fi].set_ylabel('')
            axes[bi, fi].set_xticklabels(axes[bi, fi].get_xticklabels(), rotation=90)
        while fi < 6:
            fi += 1
            axes[bi, fi].set_axis_off()
    return None

if False:
    batches = [3,8,12]
    plot_percept_features_batches(batches)


def export_percept_feature_counts(perm_batch=False):
    print('Exporting feautre counts...', end='')
    batches = range(1,101)
    count = {}
    feat_name = {}
    for bi, batch in enumerate(batches):
        if not perm_batch:
            stats = pd.read_csv(os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'))
        else:
            stats = pd.read_csv(os.path.join('results_svm', 'permutation', 'svm_batch_{}'.format(batch), 'percept_features_stats.csv'))
        stats = stats[stats['p-val'] < 0.05]
        feats = np.unique(stats['index'])
        for fi, feat in enumerate(feats):
            channel, feature = interpret_feat(feat)
            if feature not in count:
                count[feature] = {}
                feat_name[feature] = {}
            if channel not in count[feature]:
                count[feature][channel] = 0
                feat_name[feature][channel] = feat
            count[feature][channel] += 1
    df_count = pd.DataFrame(count)
    df_count.reset_index(inplace = True)
    df_count.rename(columns = {'index': 'channel'}, inplace=True)
    df_count = df_count.melt(id_vars = 'channel', var_name = 'feature', value_name = 'count')
    df_count.dropna(inplace = True)

    df_feat = pd.DataFrame(feat_name)
    df_feat.reset_index(inplace = True)
    df_feat.rename(columns = {'index': 'channel'}, inplace=True)
    df_feat = df_feat.melt(id_vars = 'channel', var_name = 'feature', value_name = 'feat')
    df_feat.dropna(inplace = True)

    df = df_count.merge(df_feat, how = 'left', on = ['channel', 'feature'])
    df['n_batch'] = len(batches)
    df['likelihood'] = df['count']/len(batches)
    if not perm_batch:
        df.to_csv('summary_percept_feature_counts.csv', index=False)
    else:
        df.to_csv('summary_percept_feature_counts_permutation.csv', index=False)
    print('Done.')


def plot_percept_feature_counts(perm_counts=False):
    if not perm_counts:
        df = pd.read_csv('summary_percept_feature_counts.csv')
    else:
        df = pd.read_csv('summary_percept_feature_counts_permutation.csv')
    df.sort_values(by='likelihood', ascending=False, inplace=True)
    feats = np.unique(df.feature)
    colors = get_colors(feats)
    sns.set_theme(style="white", font_scale=1.5)
    f, axes = plt.subplots(nrows=2, ncols=4, sharey=True)
    for fi, feat in enumerate(feats):
        ri = np.floor(fi/4).astype(int)
        ci = fi - ri*4
        tp = df[df.feature == feat]
        sns.pointplot(ax=axes[ri, ci], data=tp, x='channel', y='likelihood',
                      color=colors[fi])
        axes[ri,ci].set_title(feat)
        axes[ri,ci].set_xlabel('')
        axes[ri,ci].set_ylabel('')
        axes[ri,ci].set_xticklabels(axes[ri,ci].get_xticklabels(), rotation=90)
    for ri in range(2):
        axes[ri, 0].set_ylabel('probability')
    return None


def plot_likelihood_distribution(return_feats = False):
    df_perm = pd.read_csv('summary_percept_feature_counts_permutation.csv')
    df_real = pd.read_csv('summary_percept_feature_counts.csv')
    cutoff = np.percentile(df_perm['likelihood'], q=100)
    print('Cutoff is likelihood > {:3.3f}'.format(cutoff))
    if not return_feats:
        f = plt.figure()
        plt.hist(df_perm['likelihood'], bins = 50, density=True)
        plt.hist(df_real['likelihood'], bins = 50, alpha = 0.75, density=True)
        plt.legend(['perm', 'real'])
        plt.vlines(cutoff, ymin=0, ymax=25, colors='r', linewidth = 2, label='perm 95% percentile')
        plt.xlabel('feature probability')
        plt.ylabel('density')
        plt.title('feature probability distribution')
    else:
        return df_real[df_real.likelihood > cutoff]


if False:
    export_percept_feature_counts(False)
    export_percept_feature_counts(True)
    export_percept_features()
    export_stat_percept_features()
    export_percept_motion_clock()
    export_lr_motion_percept()
    export_lr_percept_clock()


def export_percept_motion_clock():

    warnings.filterwarnings('ignore')

    stats = pd.read_csv(os.path.join('percept_features_stats.csv'))
    stats = stats[stats['p-val'] < 0.05]
    feats = stats['index'].tolist()

    f_load = 'summary_model.csv'
    df = pd.read_csv(f_load)
    df.grating = df.grating.astype(str)
    percept = []
    clock = []
    clock_600 = []
    motion = []
    p_list = []
    counti = 0
    participants = get_participants()
    for p in participants:
        for mi, m in enumerate([0,5,10,20]):
            [X1, flabels] = get_fooof_features('st', p, m, feats_use = feats, verbose = False)
            [X0, flabels] = get_fooof_features('rs', p, m, feats_use = feats, verbose = False)
            X1[X1==0] = np.nan
            X0[X0==0] = np.nan
            X = X1-X0
            percept.append( np.nanmean(X, axis=0).flatten().tolist() )
            tp =  df.loc[ np.logical_and(df['participant'] == p, df['grating'] == str(m)), 'lambda']
            tp2 = df.loc[ np.logical_and(df['participant'] == p, df['grating'] == str(m)), 'lambda_600']
            clock.append(np.mean(tp))
            clock_600.append(np.mean(tp2))
            motion.append(m)
            p_list.append(p)
            counti += 1
            print('{:3.2f}%'.format(counti/len(participants)/4*100), end='\r')
    df = pd.DataFrame(np.array(percept), columns = flabels)
    df['motion'] = motion
    df['clock'] = clock
    df['clock_600'] = clock_600
    df['participant'] = p_list
    df['log_motion'] = df.motion
    idx = df.motion > 0
    df.loc[idx, 'log_motion'] = np.log(df.loc[idx, 'motion'])
    df.to_csv('percept_motion_clock.csv', index=False)
    print('Done.')
    return None


def export_lr_motion_percept():
    f_load = os.path.join('percept_motion_clock.csv')
    df = pd.read_csv(f_load)
    feats = df.columns[:-5].tolist()
    res_pm = []
    res_pl = []
    for feat in feats:
        tp = df[[feat, 'motion', 'log_motion']].copy()
        tp.dropna(inplace=True)

        m = pg.linear_regression(X= tp.motion, y = tp[feat])
        res_pm.append([m['r2'][1], m['pval'][1]])
        m = pg.linear_regression(X= tp.log_motion, y = tp[feat])
        res_pl.append([m['r2'][1], m['pval'][1]])
    res_pm = np.array(res_pm)
    res_pl = np.array(res_pl)
    corr = pd.DataFrame({'r2_pm': res_pm[:,0],
                         'p_pm': res_pm[:,1],
                         'r2_pl': res_pl[:,0],
                         'p_pl': res_pl[:,1],
                         'feat': feats})
    corr['channel'] = interpret_feats(corr.feat)[:,0]
    corr['feature'] = interpret_feats(corr.feat)[:,1]
    corr.sort_values(by='feature', inplace=True)
    corr.to_csv('motion_percept_lr.csv', index=False)
    return None


def export_lr_percept_clock():
    if os.path.exists('results_lr'):
        for f in os.listdir('results_lr'):
            os.remove(os.path.join('results_lr',f))

    f_load = 'percept_motion_clock.csv'
    df_raw = pd.read_csv(f_load)
    feats = df_raw.columns[:-5].tolist()
    dv = 'clock_600'

    col_names = feats.copy()
    col_names.append(dv)
    df_all = df_raw[col_names]

    X_corr = remove_outliers_mahalanobis(df_all.to_numpy())
    tp = pd.DataFrame(X_corr, columns = col_names)
    print('Included observations: N={}'.format(tp.shape[0]))
    m = pg.linear_regression(X=tp[feats].to_numpy(), y=tp[dv].to_numpy())
    m.names[1:] = feats
    m.round(3).to_csv(os.path.join('results_lr','percept_{}_lr_run_0.csv'.format(dv)), index=False)

    run = 1
    nfeat = [len(feats)]
    adj_r2 = [m.adj_r2[0]]
    while len(adj_r2) < 2 or adj_r2[-1] > adj_r2[-2]:
        print('Testing features...run={}...'.format(run))
        r2_ratio = []
        for fi, feat in enumerate(feats):
            tp = df_all.drop(feat, axis='columns')  # drop the feature
            col_names = tp.columns.tolist()
            X_corr = remove_outliers_mahalanobis(tp.to_numpy())
            tp = pd.DataFrame(X_corr, columns=col_names)
            tp_m = pg.linear_regression(X=tp.to_numpy()[:,:-1], y=tp.to_numpy()[:,-1])
            r2_ratio.append( tp_m.adj_r2[0]/adj_r2[-1] )
        r2_ratio = pd.DataFrame({'feat': feats,
                                 'r2_ratio':  r2_ratio})
        r2_ratio.to_csv(os.path.join('results_lr', 'test_features_run_{}.csv'.format(run)), index=False)

        rows = r2_ratio.r2_ratio < 1
        feats = r2_ratio.loc[rows, 'feat'].tolist()
        col_names = feats.copy()
        col_names.append(dv)
        df_all = df_raw[col_names]

        X_corr = remove_outliers_mahalanobis(df_all.to_numpy())
        tp = pd.DataFrame(X_corr, columns=col_names)
        print('Included observations: N={}'.format(tp.shape[0]))
        m = pg.linear_regression(X=tp[feats].to_numpy(), y=tp[dv].to_numpy())
        m.names[1:] = feats
        m.round(3).to_csv(os.path.join('results_lr', 'percept_{}_lr_run_{}.csv'.format(dv, run)), index=False)

        run += 1
        nfeat.append(len(feats))
        adj_r2.append(m.adj_r2[0])
    print('Done.')
    nfeat = nfeat[:-1]
    adj_r2 = adj_r2[:-1]
    os.remove(os.path.join('results_lr', 'percept_{}_lr_run_{}.csv'.format(dv, run-1)))
    os.remove(os.path.join('results_lr', 'test_features_run_{}.csv'.format(run-1)))
    print('#feature converged at {}'.format(nfeat[-1]))
    print('Final model:')
    m = pd.read_csv(os.path.join('results_lr', 'percept_{}_lr_run_{}.csv'.format(dv, run-2)))
    print(m)
    m.to_csv('percept_{}_lr.csv'.format(dv), index=False)

    return None


def plot_corr(feat, pair, batch = None, ax=None, color=None):
    # pair ['pc', 'mp']
    if color is None:
        color = '#0000FF'
    if batch is None:
        f_load = 'percept_motion_clock.csv'
    else:
        f_load = os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_motion_clock.csv')
    df = pd.read_csv(f_load)
    if pair == 'pc':
        dv = 'clock_600'
        # multivariate outliers
        feats = pd.read_csv('percept_{}_lr.csv'.format(dv)).names.tolist()[1:]
        colnames = feats.copy()
        colnames.append(dv)
        X_corr = remove_outliers_mahalanobis(df[colnames].to_numpy())
        df = pd.DataFrame(X_corr, columns = colnames)

        x = df[feat]
        y = df[dv]
    elif pair == 'mp':
        x = df['log_motion']
        y = df[feat]

    if ax is None:
        f = plt.figure()
        sns.regplot(x=x,y=y,color=color)
        return f
    else:
        sns.regplot(x=x,y=y,ax=ax, color=color)
        return ax


def examine_clock_features(batch = None, return_stats = False):
    alpha = 0.05
    if batch is None:
        f_load = 'percept_clock_600_lr.csv'
    else:
        f_load = os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'percept_clock_600_lr.csv')
    stats = pd.read_csv(f_load)
    feats  = stats[stats['pval'] < alpha]
    if return_stats:
        return feats
    print(feats)
    rows, cols = 1,4  # layout
    sns.set_theme(style="white", font_scale=2)
    f, axes = plt.subplots(rows, cols)
    for fi, feati in enumerate( range(1,feats.shape[0]) ):  # skip the intercept
        ri = np.floor(fi/cols).astype(int)
        ci = int(fi - ri*cols)
        if rows == 1:
            ax = axes[ci]
        else:
            ax = axes[ri, ci]
        feat = feats.iloc[feati]['names']
        chan, feature = interpret_feat(feat, True)
        res = plot_corr(feat, 'pc',  batch=batch, ax=ax, color=get_colors(feature))
        ax.set_title('{} {}'.format(chan, feature))
        ax.set_xlabel('( stimulus on - stimulus off )')
        if ci == 0:
            ax.set_ylabel('clock speed (λ600)')
        else:
            ax.set_ylabel('')
        print('{} {}: β={:3.3f}, p={:3.3f}'.format(chan, feature, feats.iloc[feati]['coef'], feats.iloc[feati]['pval']))
    return None



def examine_motion_percept(batch = None):
    alpha = 0.05
    if batch is None:
        f_load = 'motion_percept_lr.csv'
    else:
        f_load = os.path.join('results_svm', 'svm_batch_{}'.format(batch), 'motion_percept_lr.csv')
    stats = pd.read_csv(f_load)
    feats  = stats[stats['p_pl'] < alpha]  # log motion
    print(feats)
    rows, cols = 1,4  # layout
    sns.set_theme(style="white", font_scale=2)
    f, axes = plt.subplots(rows, cols)
    for fi, feati in enumerate( range(feats.shape[0]) ):  # skip the intercept
        ri = np.floor(fi/cols).astype(int)
        ci = int(fi - ri*cols)
        if rows == 1:
            ax = axes[ci]
        else:
            ax = axes[ri, ci]
        feat = feats.iloc[feati]['feat']
        chan, feature = interpret_feat(feat, True)
        res = plot_corr(feat, 'mp',  batch=batch, ax=ax, color=get_colors(feature))
        ax.set_title('{} {}'.format(chan, feature))
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([0, np.log(5), np.log(10), np.log(20)])
        ax.set_xticklabels([0, 5, 10, 20])
        print('{} {}: R2={:3.3f}, p={:3.3f}'.format(chan, feature, feats.iloc[feati]['r2_pl'], feats.iloc[feati]['p_pl']))
    return None


def remove_outliers_mahalanobis(X, keep_shape=False):
    from scipy.stats import chi2
    # X - array (n_case, n_feature)
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)

    X_hat = np.tile(np.nanmean(X, axis=0, keepdims=True), (X.shape[0], 1))
    nas = np.isnan(X)
    X[nas] = X_hat[nas] # replace nan with mean
    S = np.cov(X.transpose())
    d2 = ((X - X_hat) @ np.linalg.inv(S) @ (X-X_hat).transpose()).diagonal()
    d = np.sqrt(d2)

    k = X.shape[1]
    alpha = 0.05
    ck = chi2.ppf(1-alpha, df=k)**0.5  # ppf is inverse of cdf

    X_corr = X.copy()
    if keep_shape:
        X_corr[d>ck,:] = np.nan
    else:
        X_corr = X[d<=ck,:].copy()
    return X_corr

