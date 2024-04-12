# To use this script, a connection to ACR-T needs to established first.
# Correspondance: cyj.sci@gmail.com

import actr
import numbers
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from itertools import product

#matplotlib.use('TkAgg')

realTimeOn = False

lf_de = 0.4
lf_in = 0.2

def get_participants():
    p_list = list(range(201,231))
    p_list.extend(list(range(232, 242)))
    return p_list

best_fit_criteria = ['HR', 'CR']

def tick_number_sims(a, b):
    if isinstance(b, numbers.Number) and isinstance(a, numbers.Number):
        return -abs(a - b) * 0.1
    else:
        return False

if False:
    actr.add_command("tick-number-sims", tick_number_sims,
                     "Similarity between ticks for the temporal oddball task.")


actr.load_act_r_model("ACT-R:Proj_TD;oddball-v6-no-adapt.lisp")
response = False
response_time = False


def respond_to_key_press(model, key):
    # store the key that was pressed in the response variable
    # call the AGI command that clears the window

    global response, response_time
    response_time = actr.get_time(model)
    response = key

    actr.clear_exp_window()


def load_beh(participant, block='', grating_freq=None):
    columns = ['direction', 'duration', 'trigger', 'stim_type', 'grating_freq',
               'corr_ans', 'resp.rt', 'resp.keys', 'resp.corr']
    if participant >200:
        p_beh = os.path.join('data', 'beh', '{}_oddball_raw.csv'.format(participant))
    else:
        p_beh = os.path.join('data', 'beh', '{}_oddball.csv'.format(participant))
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


def best_fit(data, criteria, x, weights=None):
    if type(criteria)==list and len(criteria)>1:
        if weights is not None:
            data[criteria] = data[criteria] * weights
        else:
            data[criteria] = data[criteria] * 1/len(criteria)
        tp = np.sum(data[criteria], axis=1).tolist()
    else:
        tp = data[criteria].tolist()
    x_list = data[x].tolist()
    id = tp.index(min(tp))
    return x_list[id], min(tp)


class block():
    def __init__(self, condition):
        block.name = condition
    def choose_goal(self, alpha, beta):
        if self.name == 'decreament':
            if beta is None:
                actr.goal_focus('gd{}'.format(alpha))
            else:
                actr.goal_focus('gd{}-b{}'.format(alpha, beta))
            actr.set_parameter_value(':lf', lf_de)
            # return True
        elif self.name == 'increament':
            if beta is None:
                actr.goal_focus('gi{}'.format(alpha))
            else:
                actr.goal_focus('gi{}-b{}'.format(alpha, beta))
            actr.set_parameter_value(':lf', lf_in)
            # return True


def trial(duration):
    window = actr.open_exp_window("Oddball", visible=realTimeOn)
    actr.add_text_to_exp_window(window, 'V', x=125, y=150)
    actr.install_device(window)

    actr.schedule_event_relative(duration, "clear-exp-window",
                                 params=[window],
                                 time_in_ms=True)

    actr.add_command("oddball-key-press", respond_to_key_press,
                     "Oddball task output-key monitor")
    actr.monitor_command("output-key", "oddball-key-press")

    global response
    response = False
    start = actr.get_time()

    actr.run(3, realTimeOn)
    rt = response_time - start - duration

    actr.remove_command_monitor("output-key", "oddball-key-press")
    actr.remove_command("oddball-key-press")

    return response, rt


def get_params(participant, f_params):
    #print('Reading fitted params...', end='')
    df = pd.read_csv(f_params)
    alpha_frex = {}
    iclock_frex = {}
    alpha_init = {}
    for freq in [0, 5, 10, 20]:
        rows = np.logical_and(df.participant == participant, df.grating == freq)
        row_d = np.logical_and(rows, df.direction == 'decrement')
        alpha_d = df.loc[row_d, 'A'].tolist()[0]
        row_i = np.logical_and(rows, df.direction == 'increment')
        alpha_i = df.loc[row_i, 'A'].tolist()[0]
        a = df.loc[row_i, 'a'].tolist()[0]
        start = df.loc[row_i, 'start'].tolist()[0]
        alpha = {'decreament': alpha_d, 'increament': alpha_i}
        iclock = {'time-mult': a, 'start': start}
        alpha_frex[freq] = alpha
        iclock_frex[freq] = iclock
    #print('Done.')
    alpha_init['decreament'] = df.loc[row_d, 'A0'].tolist()[0]
    alpha_init['increament'] = df.loc[row_i, 'A0'].tolist()[0]
    beta = df.loc[row_i,'B'].tolist()[0]
    n_step = df.loc[row_i,'n_step'].tolist()[0]
    params = {'alpha_init': alpha_init,
              'alpha_frex': alpha_frex,
              'iclock_frex': iclock_frex,
              'beta': beta,
              'n_step': n_step}
    return params


def nonadapt(participant, alpha=None, beta= None, load_fit=False,
             block_type='',
             start_increment=0.011,
             time_mult=1.1, verbose = False,
             return_resp = False):
    # alpha could be an integer or a dict e.g., {'increament': 1, 'decreament':3}
    if load_fit:
        load_params = True
        params = get_params(participant, 'fitting_nonadapt.csv')
        alpha_frex = params['alpha_frex']
        iclock_frex = params['iclock_frex']
        beta = params['beta']
    else:
        load_params = False

    if beta is not None:
        if np.isnan(beta) or beta==0:
            beta = None
        else:
            beta = int(beta)

    actr.reset()

    beh = load_beh(participant, block_type)
    blocks = beh.direction.tolist()
    gratings = beh.grating_freq.tolist()
    durations = beh.duration*1000
    durations = [int(x) for x in durations]
    correct_answers = beh.corr_ans.tolist()
    human_responses = beh['resp.keys'].tolist()
    responses = []
    rts = []
    hit = 0
    corr_rejection = 0
    human_hit = 0
    human_cr = 0
    n_target = len([x for x in correct_answers if x=='j'])
    n_nontarget = len(correct_answers)-n_target
    last_block = blocks[0]
    last_freq = gratings[0]

    if load_params:
        block(last_block).choose_goal(alpha=int(alpha_frex[last_freq][last_block]), beta=beta)
        actr.set_parameter_value(':time-master-start-increment', iclock_frex[last_freq]['start'])
        actr.set_parameter_value(':time-mult', iclock_frex[last_freq]['time-mult'])
    else:
        if type(alpha)==dict:
            block(last_block).choose_goal(alpha=int(alpha[last_block]), beta=beta)
        else:
            block(last_block).choose_goal(alpha=int(alpha), beta=beta)
        actr.set_parameter_value(':time-master-start-increment', start_increment)
        actr.set_parameter_value(':time-mult', time_mult)

    for di, duration in enumerate(durations):
        current_block = blocks[di]
        current_freq = gratings[di]
        if current_block != last_block or current_freq != last_freq:  # if block changes
            if load_params:
                block(current_block).choose_goal(alpha=int(alpha_frex[current_freq][current_block]), beta=beta)
            else:
                if type(alpha) == dict:
                    block(current_block).choose_goal(alpha=int(alpha[current_block]), beta=beta)
                else:
                    block(current_block).choose_goal(alpha=int(alpha), beta=beta)

        if load_params and current_freq != last_freq:
            actr.set_parameter_value(':time-master-start-increment', iclock_frex[current_freq]['start'])
            actr.set_parameter_value(':time-mult', iclock_frex[current_freq]['time-mult'])

        last_block = current_block
        last_freq = current_freq
        response, rt = trial(duration)

        if response:
            rts.append(rt)
        else:
            response = 'None'
        responses.append(response)

        corr_ans = correct_answers[di]
        human_response = human_responses[di]
        if corr_ans == 'j' and response == 'j':
            hit += 1
        if corr_ans == 'None' and response == 'None':
            corr_rejection += 1
        if corr_ans == 'j' and human_response == 'j':
            human_hit += 1
        if corr_ans == 'None' and human_response == 'None':
            human_cr += 1

        hr_adj = (hit + 0.1) / (n_target + 0.2)
        far_adj = (n_nontarget-corr_rejection + 0.9) / (n_nontarget + 1.8)
        dprime = norm.ppf(hr_adj) - norm.ppf(far_adj)

        hr_human_adj = (human_hit + 0.1) / (n_target + 0.2)
        far_human_adj = (n_nontarget-human_cr + 0.9) / (n_nontarget + 1.8)
        dprime_human = norm.ppf(hr_human_adj) - norm.ppf(far_human_adj)

    performance = [hit/n_target, human_hit/n_target,
                   corr_rejection/n_nontarget, human_cr/n_nontarget,
                   dprime, dprime_human]
    #print('Correct answers are: {}'.format(correct_answers))
    #print('Current responses are: {}'.format(responses))
    if verbose:
        print('Hit rate is {0:.3f} SIMULATION {1:.3f} HUMAN '.format(performance[0], performance[1]))
        print('Correct rejection rate is {0:.3f} SIMULATION {1:.3f} HUMAN'.format(performance[2], performance[3]))
        print('D-prime is {0:.3f} SIMULATION {1:.3f} HUMAN'.format(performance[4], performance[5]))
        print('')
    if not return_resp:
        return performance
    else:
        return responses


def adapt(participant, step=None, alpha=None, beta=None, inherit=False, load_fit=False,
          block_type = '', grating_freq=None,
          increment=0.001,
          start_increment=0.011,
          time_mult=1.1, verbose=False,
          return_resp = False): #step: in trials

    if inherit:
        load_params = True
        params = get_params(participant,'fitting_nonadapt.csv')
        alpha_frex = params['alpha_frex']
        iclock_frex = params['iclock_frex']
        beta = params['beta']

    elif load_fit:
        load_params = True
        params = get_params(participant, 'fitting_adapt.csv')
        alpha_frex = params['alpha_frex']
        iclock_frex = params['iclock_frex']
        beta = params['beta']
        step = params['n_step']
    else:
        load_params = False

    if beta is not None:
        if np.isnan(beta) or beta==0:
            beta = None
        else:
            beta = int(beta)

    actr.reset()

    beh = load_beh(participant, block=block_type, grating_freq=grating_freq)
    blocks = beh.direction.tolist()
    gratings = beh.grating_freq.tolist()
    durations = beh.duration*1000
    durations = [int(x) for x in durations]
    correct_answers = beh.corr_ans.tolist()
    human_responses = beh['resp.keys'].tolist()
    responses = []
    rts = []
    hit = 0
    corr_rejection = 0
    human_hit = 0
    human_cr = 0
    n_target = len([x for x in correct_answers if x=='j'])
    n_nontarget = len(correct_answers)-n_target
    last_block = blocks[0]
    last_freq = gratings[0]

    if load_params:
        block(last_block).choose_goal(alpha=int(alpha_frex[last_freq][last_block]), beta=beta)
        time_master_start_increment = iclock_frex[last_freq]['start']
        actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
        actr.set_parameter_value(':time-mult', iclock_frex[last_freq]['time-mult'])
    else:
        if type(alpha)==dict:
            block(last_block).choose_goal(alpha=int(alpha[last_block]), beta=beta)
        else:
            block(last_block).choose_goal(alpha=int(alpha), beta=beta)
        time_master_start_increment = start_increment
        actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
        actr.set_parameter_value(':time-mult', time_mult)

    count = 0
    for di, duration in enumerate(durations):
        current_block = blocks[di]
        current_freq = gratings[di]
        if current_block != last_block or current_freq != last_freq:  # if block changes
            if load_params:
                block(current_block).choose_goal(alpha=int(alpha_frex[current_freq][current_block]), beta=beta)
            else:
                if type(alpha) == dict:
                    block(current_block).choose_goal(alpha=int(alpha[current_block]), beta=beta)
                else:
                    block(current_block).choose_goal(alpha=int(alpha), beta=beta)
            count = 0

        if load_params and current_freq != last_freq:
            time_master_start_increment = iclock_frex[current_freq]['start']
            actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
            actr.set_parameter_value(':time-mult', iclock_frex[current_freq]['time-mult'])

        last_block = current_block
        last_freq = current_freq
        response, rt = trial(duration)

        if response:
            rts.append(rt)
        else:
            response = 'None'
        responses.append(response)

        corr_ans = correct_answers[di]
        human_response = human_responses[di]
        if corr_ans == 'j' and response == 'j':
            hit += 1
        if corr_ans == 'None' and response == 'None':
            corr_rejection += 1
        if corr_ans == 'j' and human_response == 'j':
            human_hit += 1
        if corr_ans == 'None' and human_response == 'None':
            human_cr += 1

        hr_adj = (hit + 0.1) / (n_target + 0.2)
        far_adj = (n_nontarget - corr_rejection + 0.9) / (n_nontarget + 1.8)
        dprime = norm.ppf(hr_adj) - norm.ppf(far_adj)

        hr_human_adj = (human_hit + 0.1) / (n_target + 0.2)
        far_human_adj = (n_nontarget - human_cr + 0.9) / (n_nontarget + 1.8)
        dprime_human = norm.ppf(hr_human_adj) - norm.ppf(far_human_adj)

        count += 1
        if count == step:
            time_master_start_increment += increment
            actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
            count = 0

    performance = [hit / n_target, human_hit / n_target,
                   corr_rejection / n_nontarget, human_cr / n_nontarget,
                   dprime, dprime_human]
    #print('Correct answers are: {}'.format(correct_answers))
    #print('Current responses are: {}'.format(responses))
    if verbose:
        print('Hit rate is {0:.3f} SIMULATION {1:.3f} HUMAN '.format(performance[0], performance[1]))
        print('Correct rejection rate is {0:.3f} SIMULATION {1:.3f} HUMAN'.format(performance[2], performance[3]))
        print('D-prime is {0:.3f} SIMULATION {1:.3f} HUMAN'.format(performance[4], performance[5]))
        print('')
    if not return_resp:
        return performance
    else:
        return responses


def novel(participant, step=None, alpha=None, beta=None, inherit=False, load_fit=False,
          block_type='', grating_freq=None,
          increment=0.001,
          start_increment=0.011,
          time_mult=1.1, verbose=False,
          return_resp = False):  #step: in trials

    if inherit:
        load_params = True
        params = get_params(participant,'fitting_nonadapt.csv')
        alpha_frex = params['alpha_frex']
        iclock_frex = params['iclock_frex']
        beta = params['beta']

    elif load_fit:
        load_params = True
        params = get_params(participant, 'fitting_novel.csv')
        alpha_frex = params['alpha_frex']
        iclock_frex = params['iclock_frex']
        beta = params['beta']
        step = params['n_step']
    else:
        load_params = False

    if beta is not None:
        if np.isnan(beta) or beta==0:
            beta = None
        else:
            beta = int(beta)

    actr.reset()

    beh = load_beh(participant, block=block_type, grating_freq=grating_freq)
    blocks = beh.direction.tolist()
    gratings = beh.grating_freq.tolist()
    durations = beh.duration*1000
    durations = [int(x) for x in durations]
    correct_answers = beh.corr_ans.tolist()
    human_responses = beh['resp.keys'].tolist()
    responses = []
    rts = []
    hit = 0
    corr_rejection = 0
    human_hit = 0
    human_cr = 0
    n_target = len([x for x in correct_answers if x=='j'])
    n_nontarget = len(correct_answers)-n_target
    last_block = blocks[0]
    last_freq = gratings[0]

    if load_params:
        block(last_block).choose_goal(alpha=int(alpha_frex[last_freq][last_block]), beta=beta)
        time_master_start_increment = iclock_frex[last_freq]['start']
        actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
        actr.set_parameter_value(':time-mult', iclock_frex[last_freq]['time-mult'])
    else:
        if type(alpha)==dict:
            block(last_block).choose_goal(alpha=int(alpha[last_block]), beta=beta)
        else:
            block(last_block).choose_goal(alpha=int(alpha), beta=beta)
        time_master_start_increment = start_increment
        actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
        actr.set_parameter_value(':time-mult', time_mult)

    count = 0
    for di,duration in enumerate(durations):

        current_block = blocks[di]
        current_freq = gratings[di]

        if current_block != last_block or current_freq != last_freq:  # if block changes
            if load_params:
                block(current_block).choose_goal(alpha=int(alpha_frex[current_freq][current_block]), beta=beta)
            else:
                if type(alpha) == dict:
                    block(current_block).choose_goal(alpha=int(alpha[current_block]), beta=beta)
                else:
                    block(current_block).choose_goal(alpha=int(alpha), beta=beta)
            count = 0

        if load_params and current_freq != last_freq:
            time_master_start_increment = iclock_frex[current_freq]['start']
            actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
            actr.set_parameter_value(':time-mult', iclock_frex[current_freq]['time-mult'])

        last_block = current_block
        last_freq = current_freq
        response, rt = trial(duration)

        count += 1
        if response:
            rts.append(rt)
            if load_params:
                time_master_start_increment = iclock_frex[current_freq]['start']
            else:
                time_master_start_increment = start_increment
            actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
            count = 0
        else:
            response = 'None'
            if count == step:
                time_master_start_increment += increment
                actr.set_parameter_value(':time-master-start-increment', time_master_start_increment)
                count = 0
        responses.append(response)

        corr_ans = correct_answers[di]
        human_response = human_responses[di]
        if corr_ans == 'j' and response == 'j':
            hit += 1
        if corr_ans == 'None' and response == 'None':
            corr_rejection += 1
        if corr_ans == 'j' and human_response == 'j':
            human_hit += 1
        if corr_ans == 'None' and human_response == 'None':
            human_cr += 1

        hr_adj = (hit + 0.1) / (n_target + 0.2)
        far_adj = (n_nontarget - corr_rejection + 0.9) / (n_nontarget + 1.8)
        dprime = norm.ppf(hr_adj) - norm.ppf(far_adj)

        hr_human_adj = (human_hit + 0.1) / (n_target + 0.2)
        far_human_adj = (n_nontarget - human_cr + 0.9) / (n_nontarget + 1.8)
        dprime_human = norm.ppf(hr_human_adj) - norm.ppf(far_human_adj)

    performance = [hit / n_target, human_hit / n_target,
                   corr_rejection / n_nontarget, human_cr / n_nontarget,
                   dprime, dprime_human]
    #print('Correct answers are: {}'.format(correct_answers))
    #print('Current responses are: {}'.format(responses))
    if verbose:
        print('Hit rate is {0:.3f} SIMULATION {1:.3f} HUMAN '.format(performance[0], performance[1]))
        print('Correct rejection rate is {0:.3f} SIMULATION {1:.3f} HUMAN'.format(performance[2], performance[3]))
        print('D-prime is {0:.3f} SIMULATION {1:.3f} HUMAN'.format(performance[4], performance[5]))
        print('')
    if not return_resp:
        return performance
    else:
        return responses

def tune_alpha_init(participant, model_name, block_type, f_output,
               metric_weights=None, plotOn=False, verbose=False):
    a_list = range(0,9)
    output = pd.read_csv(f_output)
    n_step = get_params(participant, f_output)['n_step']

    rows = np.logical_and(output['participant'] == participant, output['direction'] == block_type)
    print('Tuning alpha...{}...'.format(block_type))

    RMSE = []
    for i, a in enumerate(a_list):
        if verbose: print('Trying alpha={}'.format(a))
        if model_name == 'nonadapt':
            p = nonadapt(participant, block_type=block_type, alpha=a, beta=None, verbose=verbose)
        elif model_name == 'adapt':
            p = adapt(participant, step=n_step, block_type=block_type, alpha=a, beta=None, verbose=verbose)
        elif model_name == 'novel':
            p = novel(participant, step=n_step, block_type=block_type, alpha=a, beta=None, verbose=verbose)

        RMSE.append([np.square(p[0] - p[1]),
                     np.square(p[2] - p[3]),
                     np.square(p[4] - p[5])])  #d'
        print('{:3.2f}%'.format((i+1)/len(a_list)*100), end='\r')
    RMSE = np.array(RMSE)
    data = pd.DataFrame({'alpha': a_list,
                         'HR': RMSE[:,0],
                         'CR': RMSE[:,1],
                         'd-prime': RMSE[:,2]})
    data.to_csv('temp_data.csv')

    data = pd.read_csv('temp_data.csv')
    fit_x, min_error = best_fit(data=data.copy(), x = 'alpha',
                                criteria=best_fit_criteria, weights=metric_weights)
    if plotOn:
        data = data.melt(id_vars = ['alpha'], value_vars=['HR', 'CR', 'd-prime'],
                         var_name = 'Performance', value_name = 'RMSE')
        ax = sns.lineplot(data=data, x='alpha', y='RMSE', hue='Performance')
        ax.scatter(x=fit_x, y=min_error, color='r')
        ax.set(title = 'Best Fit Is Step {}'.format(fit_x))
        plt.show()
    print('Best Fit Is Alpha {}'.format(fit_x))

    output.loc[rows, 'A0'] = fit_x
    output.loc[rows, 'A'] = fit_x
    output.to_csv(f_output, index=False)


def tune_internal_clock(participant, model_name, grating_freq, f_output, metric_weights=None, verbose=False):
    time_mult_rng = [x/100 for x in range(101,130,1)]
    start_increment_rng = [x/1000 for x in range(6, 16)]

    output = pd.read_csv(f_output)
    row = np.logical_and(output.participant == participant, output.grating == grating_freq)

    RMSE = []
    print('Tuning a...{}Hz...'.format(grating_freq))
    for i, t in enumerate(time_mult_rng):
        if verbose: print('Trying time-mult={}'.format(t))
        output.loc[row,'a'] = t
        output.to_csv(f_output, index=False)
        if model_name == 'nonadapt':
            p = nonadapt(participant=participant, load_fit=True, verbose=verbose)
        elif model_name == 'adapt':
            p = adapt(participant=participant, load_fit=True, verbose=verbose)
        elif model_name == 'novel':
            p = novel(participant=participant, load_fit=True, verbose=verbose)
        RMSE.append([np.square(p[0] - p[1]), np.square(p[2] - p[3]), np.square(p[4] - p[5])])
        print('{:3.2f}%'.format((i+1)/len(time_mult_rng)*100), end='\r')
    RMSE = np.array(RMSE)
    data = pd.DataFrame({'time_mult': time_mult_rng,
                         'HR': RMSE[:, 0],
                         'CR': RMSE[:, 1],
                         'd-prime': RMSE[:, 2]})
    fit_x, min_error = best_fit(data.copy(), x='time_mult',
                                criteria=best_fit_criteria, weights=metric_weights)
    print('Initial Fit Is TIME-MULT {}'.format(fit_x))
    output.loc[row, 'a'] = fit_x
    output.to_csv(f_output, index=False)
    fit_t = fit_x

    RMSE = []
    print('Tuning start...{}Hz...'.format(grating_freq))
    for i, t0 in enumerate(start_increment_rng):
        if verbose: print('Trying start={}'.format(t0))
        output.loc[row,'start'] = t0
        output.to_csv(f_output, index=False)
        if model_name == 'nonadapt':
            p = nonadapt(participant=participant, load_fit=True, verbose=verbose)
        elif model_name == 'adapt':
            p = adapt(participant=participant, load_fit=True, verbose=verbose)
        elif model_name == 'novel':
            p = novel(participant=participant, load_fit=True, verbose=verbose)
        RMSE.append([np.square(p[0] - p[1]), np.square(p[2] - p[3]), np.square(p[4] - p[5])])
        print('{:3.2f}%'.format((i + 1) / len(start_increment_rng) * 100), end='\r')
    RMSE = np.array(RMSE)
    data = pd.DataFrame({'start_increment': start_increment_rng,
                         'HR': RMSE[:, 0],
                         'CR': RMSE[:, 1],
                         'd-prime': RMSE[:, 2]})
    fit_x, min_error = best_fit(data.copy(), x='start_increment',
                                criteria=best_fit_criteria, weights=metric_weights)
    print('Best Fit Is START-INCREMENT {}'.format(fit_x))
    output.loc[row, 'start'] = fit_x
    output.to_csv(f_output, index=False)

    # refit t with fitted t0
    t_fit_rng = [fit_t-0.03, fit_t-0.02, fit_t-0.01, fit_t, fit_t+0.01, fit_t+0.02, fit_t+0.03]
    RMSE = []
    print('Fine-tuning a...{}Hz...'.format(grating_freq))
    for i,t in enumerate(t_fit_rng):
        if verbose: print('Trying time-mult={}'.format(t))
        output.loc[row,'a'] = t
        output.to_csv(f_output, index=False)
        if model_name == 'nonadapt':
            p = nonadapt(participant=participant, load_fit=True, verbose=verbose)
        elif model_name == 'adapt':
            p = adapt(participant=participant, load_fit=True, verbose=verbose)
        elif model_name == 'novel':
            p = novel(participant=participant, load_fit=True, verbose=verbose)
        RMSE.append([np.square(p[0] - p[1]), np.square(p[2] - p[3]), np.square(p[4] - p[5])])
        print('{:3.2f}%'.format((i + 1) / len(time_mult_rng) * 100), end='\r')
    RMSE = np.array(RMSE)
    data = pd.DataFrame({'time_mult': t_fit_rng,
                         'HR': RMSE[:, 0],
                         'CR': RMSE[:, 1],
                         'd-prime': RMSE[:, 2]})
    fit_x, min_error = best_fit(data.copy(), x='time_mult',
                                criteria=best_fit_criteria, weights=metric_weights)
    print('Best Fit Is TIME-MULT {}'.format(fit_x))
    output.loc[row, 'a'] = fit_x
    output.to_csv(f_output, index=False)


def tune_alpha_frex(participant, model_name, block_type, grating_freq, f_output,
                    metric_weights=None, plotOn=False, verbose=False):
    a_list = range(0,9)
    output = pd.read_csv(f_output)

    rows = np.logical_and(output['participant'] == participant, output['direction'] == block_type)
    rows = np.logical_and(rows,  output['grating'] == grating_freq)
    print('Tuning alpha...{}: {}Hz...'.format(block_type, grating_freq))

    RMSE = []
    for i, a in enumerate(a_list):
        if verbose: print('Trying alpha={}'.format(a))
        output.loc[rows, 'A'] = a
        output.to_csv(f_output, index=False)
        if model_name == 'nonadapt':
            p = nonadapt(participant, load_fit=True, verbose=verbose)
        elif model_name == 'adapt':
            p = adapt(participant, load_fit=True, verbose=verbose)
        elif model_name == 'novel':
            p = novel(participant, load_fit=True, verbose=verbose)

        RMSE.append([np.square(p[0] - p[1]),
                     np.square(p[2] - p[3]),
                     np.square(p[4] - p[5])])  #d'
        print('{:3.2f}%'.format((i+1)/len(a_list)*100), end='\r')
    RMSE = np.array(RMSE)
    data = pd.DataFrame({'alpha': a_list,
                         'HR': RMSE[:,0],
                         'CR': RMSE[:,1],
                         'd-prime': RMSE[:,2]})
    data.to_csv('temp_data.csv')

    data = pd.read_csv('temp_data.csv')
    fit_x, min_error = best_fit(data=data.copy(), x = 'alpha',
                                criteria=best_fit_criteria, weights=metric_weights)
    if plotOn:
        data = data.melt(id_vars = ['alpha'], value_vars=['HR', 'CR', 'd-prime'],
                         var_name = 'Performance', value_name = 'RMSE')
        ax = sns.lineplot(data=data, x='alpha', y='RMSE', hue='Performance')
        ax.scatter(x=fit_x, y=min_error, color='r')
        ax.set(title = 'Best Fit Is Step {}'.format(fit_x))
        plt.show()
    print('Best Fit Is Alpha {}'.format(fit_x))

    output.loc[rows, 'A'] = fit_x
    output.to_csv(f_output, index=False)


def tune_beta(participant, model_name, f_output, metric_weights=None, plotOn=False, verbose=False):
    b_list = [0]
    b_list.extend(list(range(5,17)))
    output = pd.read_csv(f_output)
    rows = output.participant == participant

    RMSE = []
    print('Tuning B...')
    for i, b in enumerate(b_list):
        if verbose: print('Trying B={}'.format(b))
        output.loc[rows,'B'] = b
        output.to_csv(f_output, index=False)
        if model_name == 'nonadapt':
            p = nonadapt(participant=participant, load_fit=True, verbose=verbose)
        elif model_name == 'adapt':
            p = adapt(participant=participant,  load_fit=True, verbose=verbose)
        elif model_name == 'novel':
            p = novel(participant=participant,  load_fit=True, verbose=verbose)
        RMSE.append([np.square(p[0] - p[1]), np.square(p[2] - p[3]), np.square(p[4] - p[5])])
        print('{:3.2f}%'.format((i+1)/len(b_list)*100), end='\r')
    RMSE = np.array(RMSE)
    data = pd.DataFrame({'beta': b_list,
                         'HR': RMSE[:,0],
                         'CR': RMSE[:,1],
                         'd-prime': RMSE[:,2]})
    data.to_csv('temp_data.csv')

    data = pd.read_csv('temp_data.csv')
    fit_x, min_error = best_fit(data.copy(), x = 'beta',
                                criteria=best_fit_criteria, weights=metric_weights)
    if plotOn:
        data = data.melt(id_vars = ['beta'], value_vars=['HR', 'CR', 'd-prime'],
                         var_name = 'Performance', value_name = 'RMSE')
        ax = sns.lineplot(data=data, x='beta', y='RMSE', hue='Performance')
        ax.scatter(x=fit_x, y=min_error, color='r')
        ax.set(title = 'Best Fit Is Step {}'.format(fit_x))
        plt.show()
    print('Best Fit Is Beta {}'.format(fit_x))

    output.loc[rows, 'B'] = fit_x
    output.to_csv(f_output, index=False)


def tune_n_step(participant, model_name, f_output, init_fit, metric_weights=None, plotOn=False, verbose=False): # adapt, novel model
    if model_name == 'adapt':
        steps = range(1,31,1)
    elif model_name == 'novel':
        steps = range(1,31,1)

    output = pd.read_csv(f_output)
    rows = output.participant==participant
    RMSE = []
    print('Tuning n_step...')
    for i, step in enumerate(steps):
        if verbose: print('Trying step={}'.format(step))
        if init_fit:
            if model_name == 'adapt':
                p = adapt(participant=participant, step=step, inherit=True, verbose=verbose)
            elif model_name == 'novel':
                p = novel(participant=participant, step=step, inherit=True, verbose=verbose)
        else:
            output.loc[rows, 'n_step'] = step
            output.to_csv(f_output, index=False)
            if model_name == 'adapt':
                p = adapt(participant=participant, load_fit=True, verbose=verbose)
            elif model_name == 'novel':
                p = novel(participant=participant, load_fit=True, verbose=verbose)
        RMSE.append([np.square(p[0]-p[1]), np.square(p[2]-p[3]), np.square(p[4]-p[5])])
        print('{:3.2f}%'.format((i+1)/len(steps)*100), end='\r')
    RMSE = np.array(RMSE)
    data = pd.DataFrame({'step': steps,
                         'HR': RMSE[:,0],
                         'CR': RMSE[:,1],
                         'd-prime': RMSE[:,2]})
    data.to_csv('temp_data.csv')

    data = pd.read_csv('temp_data.csv')
    fit_x, min_error = best_fit(data.copy(), criteria=best_fit_criteria, weights=metric_weights, x='step')
    if plotOn:
        data = data.melt(id_vars = ['step'], value_vars=['HR', 'CR', 'd-prime'],
                         var_name = 'Performance', value_name = 'RMSE')
        ax = sns.lineplot(data=data, x='step', y='RMSE', hue='Performance')
        ax.scatter(x=fit_x, y=min_error, color='r')
        ax.set(title = 'Best Fit Is Step {}'.format(fit_x))
        plt.show()
    print('Best Fit Is Step {}'.format(fit_x))
    output.loc[rows, 'n_step'] = fit_x
    output.to_csv(f_output, index=False)


def fit_actr(model_name):
    # model_name - ['nonadapt', 'adapt', 'novel']
    # ('nonadapt' needs to fitted first since the obtained params will be used to initialize the other two models)
    # 0. [adapt, novel only]. inherit A,a,start from nonadapt model, find n_step
    # 1. find initial A per direction
    # 2. find a, start per grating
    # 3. fine-tune A per grating, direction
    # 4. find B
    # 5. [adapt, novel only]. fine-tune n_step

    initialize = True  #  start a new batch fitting or continue the previous fitting
    f_output = 'fitting_{}.csv'.format(model_name)
    metric_weights = None
    participants = get_participants()
    #participants = [219, 239]
    if initialize:
        output = pd.DataFrame(columns=['participant', 'direction', 'grating',
                                       'A0', 'a', 'start', 'A', 'B', 'n_step'])
        output[['participant','direction','grating']] = list(product(participants, ['decrement','increment'], [0,5,10,20]))
        output[['A0', 'a', 'start', 'A', 'B', 'n_step']] = [None, 1.1, 0.011, None, None, None]
        output.to_csv(f_output, index=False)
    else:
        output = pd.read_csv(f_output)
        tp = np.setdiff1d(participants, np.unique(output.participant))
        if tp.shape[0] > 0:
            print('New participants involved. Update the dataframe...', end='')
            df = pd.DataFrame(columns=output.columns)
            df[['participant', 'direction', 'grating']] = list(product(tp.tolist(), ['decrement', 'increment'], [0, 5, 10, 20]))
            df[['A0', 'a', 'start', 'A', 'B', 'n_step']] = [None, 1.1, 0.011, None, None, None]
            output = pd.concat([output, df])
            print('Done.')
        else:
            print('Erasing data of the existing participants...', end='')
            ridx = np.isin(output.participant, participants)
            output.loc[ridx,['A0', 'a', 'start', 'A', 'B', 'n_step']] = np.tile(np.expand_dims([None, 1.1, 0.011, None, None, None],axis=0),[np.sum(ridx),1])
            print('Done.')

        output.to_csv(f_output, index=False)

    for pi, participant in enumerate(participants):
        print('Model fitting to data {}/{}'.format(pi+1, len(participants)))
        if model_name != 'nonadapt': tune_n_step(participant, init_fit=True, model_name=model_name,
                                                 f_output=f_output, verbose=False)

        for block_type in ['decrement', 'increment']:
            tune_alpha_init(participant, model_name=model_name, block_type = block_type,
                            f_output=f_output, metric_weights=metric_weights, verbose=False)

        for grating_freq in [0,5,10,20]:
            tune_internal_clock(participant, grating_freq=grating_freq, model_name=model_name,
                                f_output=f_output, metric_weights=metric_weights, verbose=False)

        for block_type, grating_freq in list(product(['decrement', 'increment'], [0,5,10,20])):
            tune_alpha_frex(participant, block_type=block_type, grating_freq=grating_freq, model_name=model_name,
                            f_output=f_output, metric_weights=metric_weights, verbose=False)

        tune_beta(participant, model_name=model_name, f_output=f_output, metric_weights=metric_weights, verbose=False)

        if model_name != 'nonadapt': tune_n_step(participant, init_fit=False, model_name=model_name,
                                                 f_output=f_output, verbose=False)


def run_actr():
    # run model with fitted params and get the performance: whole task
    f_output = 'performance.csv'
    participants = get_participants()
    perf = pd.DataFrame(columns=['participant','type', 'HR', 'CR', 'd-prime'])
    model_names = ['nonadapt', 'adapt', 'novel']
    perf.to_csv(f_output, index=False)

    print('Getting the performance...')
    for pi, participant in enumerate(participants):
        for m in model_names:
            if m == 'nonadapt':
                p = nonadapt(participant, load_fit=True, verbose=True)
            elif m == 'adapt':
                p = adapt(participant, load_fit=True, verbose=True)
            elif m == 'novel':
                p = novel(participant, load_fit=True, verbose=True)

            tp = pd.DataFrame({'participant': participant,
                               'type': m,
                               'HR': p[0],
                               'CR': p[2],
                               'd-prime': p[4]}, index=[0])
            perf = pd.concat([perf, tp], ignore_index=True)
        tp = pd.DataFrame({'participant': participant,
                           'type': 'human',
                           'HR': p[1],
                           'CR': p[3],
                           'd-prime': p[5]}, index=[0])
        perf = pd.concat([perf, tp], ignore_index=True)
        perf.to_csv(f_output, index=False)
        print('{:3.2f}%'.format((pi+1)/len(participants)*100), end='\r')


def get_simulated_performance():
    participants = get_participants()
    model_names = ['nonadapt', 'adapt', 'novel']
    for pi, p in enumerate(participants):
        for mi, m in enumerate(model_names):
            if m == 'nonadapt':
                resp = nonadapt(p, load_fit=True, verbose=True, return_resp=True)
            elif m == 'adapt':
                resp = adapt(p, load_fit=True, verbose=True, return_resp=True)
            elif m == 'novel':
                resp = novel(p, load_fit=True, verbose=True, return_resp=True)
            if mi == 0:
                d = []
            d.append(resp)
        pd.DataFrame( np.array(d).transpose(), columns = model_names ).to_csv( os.path.join('simulated_beh', '{}.csv'.format(p)) )


if False:
    df1 = pd.read_csv(os.path.join('oddball_v1_fit2', 'performance.csv'))
    df0 = pd.read_csv(os.path.join('oddball_v1_fit1', 'performance.csv'))
    df1.equals(df0)
    df1.compare(df0)


def get_utility(participant, model_name): # adapt, novel model
    if model_name == 'nonadapt':
        _ = nonadapt(participant=participant, load_fit=True)
    elif model_name == 'adapt':
        _ = adapt(participant=participant, load_fit=True)
    elif model_name == 'novel':
        _ = novel(participant=participant, load_fit=True)

    u_compare = actr.spp('encode-finish-compare',':u')[0][0]
    u_recall = actr.spp('encode-finish-recall',':u')[0][0]
    u_guess = actr.spp('encode-finish-guess', ':u')[0][0]

    return [u_compare, u_recall, u_guess]
