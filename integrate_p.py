""" Trying to find out whether PT and actinf are compatible or something"""
from itertools import product as prd
import ipdb

import numpy as np
from scipy import stats
import scipy as sp
from matplotlib import pyplot as plt

SD = 5
MAX_X = 100
NUM_ACTS = 2


def this_gauss(x, mu):
    return stats.norm.pdf(x - mu, scale=SD)


def get_action_attributes(action):
    """Contains a list of the attributes for all available actions. Right now,
    it only works for 2 available actions.
    """
    if action == 0:
        return 0.8, 20
    if action == 1:
        return 0.3, 12


def transition_distribution(x, current_x, action=None, custom_action=None):
    """Returns probability of --x-- given the current state --current_x-- and
    the chosen action --action-- or --custom_action--. Note that --current_x--
    is a float, which means that this function does not take as input the full
    distribution of the current state.

    Parameters
    ----------
    x : float
        Point at which to calculate the probability.
    current_x : float
        Current state of the agent or whatever.
    action : int
        Action chosen, \in range(NUM_ACTS).
    custom_action : array-like
        If this is given, it overrides --action-- and creates a custom action,
        that is, a custom function with the given set of --p-- and --mu_plus--.
        Must be included in that order in the list, that is, [p, mu_plos].

    Returns
    -------
    state : float
    Probability density for the current state, centered around --x--.
    grad : float
    The gradient dS/dp of the state.
    """
    if action is not None and custom_action is None:
        p, mu_plus = get_action_attributes(action)
    if action is None and custom_action is not None:
        p, mu_plus = custom_action
    if action is None and custom_action is None:
        raise ValueError('Either --action-- or --custom_action-- must ' +
                         'be provided.')

    state = p * this_gauss(x, current_x) + (1 - p) * this_gauss(x, current_x +
                                                                mu_plus)
    grad = this_gauss(x, current_x) + this_gauss(x, current_x + mu_plus)

    return state, grad


def next_state(current_state=0, state_scale=None, action=0):
    """Calculates the distribution over the next state, given --action--. In
    a way, this is a generalization of transition_distribution(), which takes
    into account the full distribution of the current state.

    Parameters
    ----------
    current_state : 1D ndarray or float
    Probability distribution for the current state. If a float is provided,
    it is assumed to be a Gaussian with mean at --current_state-- and standard
    deviation equal SD. Note that normalization is not checked nor enforced.

    state_scale : 1D array
    X-axis values of the values of --current_state--. Sizes must match. If
    --current_state-- is a float, --state_scale-- need not be provided; if it
    is provided, it will be used; if not, a default of np.arange(0, 100, 0.1)
    will be used instead.

    action : int
    Chosen action, as indexed in get_action_attributes().

    Returns
    -------
    next_state : 1D ndararray
    Next state, after action --action-- has been taken. It will be the same
    length of --current_state--. It is not normalized.
    """
    if state_scale is None:
        state_scale = np.arange(0, 100, 0.1)

    if isinstance(current_state, int) or isinstance(current_state, float):
        current_state = initial_state(state_scale)

    all_states = list(prd(current_state, state_scale))
    transition_out = transition_distribution(all_states[:, 0],
                                             all_states[:, 1])
    transition_out = transition_out.reshape((len(current_state), -1))
    return(transition_out)


def initial_state(x, mu=0):
    """Creates a probability distribution for the state given --mu-- and
    returns its probability density at --x--.

    """
    norm_const = 1 - stats.norm.cdf(0, loc=mu, scale=SD)
    return this_gauss(x, mu) * heaviside(x) / norm_const


def heaviside(x):
    """Heaviside function because apparently my numpy version is not cool
    enough for this pretentious function. I mean, really, who is this heaviside
    dude anyways?
    """
    return (x >= 0) * 1


def dq_dp_p(var_x, current_ixs, action):
    """Like the other, but var."""
    norm = stats.norm.pdf

    [p, mu_plus] = action

    def s(x): return p * norm(x, loc=current_ixs, scale=SD) + \
        (1 - p) * norm(x, loc=current_ixs + mu_plus, scale=SD)

    def dsdp(x): return norm(x, loc=current_ixs, scale=SD) + \
        norm(x, loc=current_ixs + mu_plus, scale=SD)
    return - dsdp(var_x) * np.log(s(var_x))


def set_goals(shape_pars, x_range):
    """Defines goals based on --shape_pars--. Returns the value of these
    at --var_x--.

    Parameters
    ----------
    shape_pars : list
    List of the form [shape, par1, par2, ...], where --shape-- is a string \in
    {'unimodal', 'exponential'}. par1, par2, etc are then the corresponding
    parameter values:
    1.- For 'unimodal', mean, standard deviation
    2.- For 'exponential', the exponent's coefficient.

    x_range : float or array with 2 values
    Range of values to consider. In 'unimodal', this can include infinity,
    but not so for 'exponential'.
    If float, the range is taken to be [0, x_range]. If array with two values,
    the range is from the first to the second one.
    """
    if shape_pars[0] == 'unimodal':
        return _goals_gauss(shape_pars, x_range)
    elif shape_pars[0] == 'exponential':
        return _goals_exp(shape_pars, x_range)


def _goals_exp(shape_pars, x_range):
    """Returns the function that returns the value of the goals at the
    requested x_var value for the exponential shape.
    """
    kappa = shape_pars[1]

    if kappa == 0:
        norm_const = np.diff(x_range)
    else:
        norm_const = kappa * np.diff(np.exp(kappa * np.array(x_range)))

    def exp_fun(x):
        return np.exp(kappa * x) / norm_const

    return exp_fun


def _goals_gauss(shape_pars, x_range):
    """Returns the function that returns the value of the goals at the
    requested x_var value.

    Gaussian version.
    """
    mean, sd = shape_pars[1:]
    norm_const = stats.norm.cdf(x_range[1], loc=mean, scale=sd)
    norm_const -= stats.norm.cdf(x_range[0], loc=mean, scale=sd)
    return lambda x: stats.norm.pdf(x, loc=mean, scale=sd) * \
        np.logical_and(x <= x_range[1], x >= x_range[0]) / norm_const


def dq_dp_fc(var_x, current_ixs, action, goals_fun):
    """Calculates dSdp * log(S)"""
    norm = stats.norm.pdf

    mu_plus = action[-1]

    def dsdp(x): return norm(x, loc=current_ixs, scale=SD) + \
        norm(x, loc=current_ixs + mu_plus, scale=SD)

    return dsdp(var_x) * goals_fun(var_x)


def dq_dp(current_ixs, action, shape_pars, x_range):
    """Calculates the final value for dQ/dp, for the probability p given in
    --action--.
    """
    goals = set_goals(shape_pars, x_range)

    def integrand_with_p(val_x):
        """Integrand function with given action."""
        return dq_dp_p(val_x, current_ixs, action)

    def integrand_with_fc(val_x):
        """Integrand function with given action."""
        return dq_dp_fc(val_x, current_ixs, action, goals)

    int_fc = sp.integrate.quad(integrand_with_p, *x_range)[0]
    int_p = sp.integrate.quad(integrand_with_fc, *x_range)[0]

    return int_fc + int_p

def current_state(var_x, action, current_ixs):
    """Calculates the currently projected state S_\tau."""
    return action[0] * this_gauss(var_x, current_ixs) + \
        (1 - action[0]) * this_gauss(var_x, current_ixs + action[1])


def dq_dmu_fun(var_x, current_ixs, action, goals_fun):
    """Returns function for the integrand in dq_dmu."""
    ds_dmu = - (1 - action[0]) * (var_x - current_ixs - action[1]) / \
            np.sqrt(2 * np.pi) / SD * this_gauss(var_x,
                                                 current_ixs + action[1])
    c_state = current_state(var_x, action, current_ixs)
    return ds_dmu * np.log(c_state) + ds_dmu - \
        ds_dmu * np.log(goals_fun(var_x))


def dq_dmu(current_ixs, action, shape_pars, x_range):
    """Calculates the final value for dQ/d(value), the value of the reward
    given in --action--.
    """
    goals = set_goals(shape_pars, x_range)

    def integrand_fun(var_x):
        """Integrand for integrating. Simple wrapper."""
        return dq_dmu_fun(var_x, current_ixs, action, goals)

    return  sp.integrate.quad(integrand_fun, *x_range)


def loop_dq_dmu(actions=None, current_ixs=0, shape_pars=None, x_range=None,
                fignum=1):
    """Calculates and plots dq_dmu for the given actions.

    Parameters
    ----------
    actions : 2D-ndarray
    2D array whose rows are [probability, reward]. Defaults to all probs of
    0.5 and rewards from 0 to 30.
    """

    if actions is None:
        num_actions = 30
        actions = np.hstack([0 * np.ones((num_actions, 1)),
                             np.arange(num_actions).reshape((-1, 1))])
    else:
        num_actions = actions.shape[0]

    results = np.inf * np.ones(num_actions)
    for ix_action in range(num_actions):
        results[ix_action] = dq_dmu(current_ixs, actions[ix_action, :],
                                    shape_pars, x_range)[0]

    fig = plt.figure(fignum)
    fig.clear()
    maax = fig.add_subplot(111)
    maax.plot(actions[:, 1], results)
    plt.draw()
    plt.show(block=False)

    return results

def kld(p, q):
    """Kullback-Leibler divergence between discrete --p-- and --q--."""
    p = np.array(p)
    q = np.array(q)
    return (p * np.log(q / p)).sum()


def integrate_kld(actions, current_ixs=0, shape_pars=None,
                  x_range=None, fignum=2):
    """calculates Q(action) for action in --actions--."""

    if actions is None:
        num_actions = 30
        actions = np.hstack([0 * np.ones((num_actions, 1)),
                             np.arange(num_actions).reshape((-1, 1))])
    else:
        num_actions = actions.shape[0]
    
    all_x = np.arange(*x_range, 0.01)
    goals = set_goals(shape_pars, x_range)(all_x)

    q_action = np.zeros(num_actions)
    for ix_action, action in enumerate(actions):
        c_state = current_state(all_x, action, current_ixs)
        q_action[ix_action] = kld(c_state, goals)

    plt.figure(fignum)
    plt.clf()
    plt.plot(q_action)
    plt.draw()
    plt.show(block=False)


    
