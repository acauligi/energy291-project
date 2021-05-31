import pdb

import gurobipy
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

class Problem():
    def __init__(self, N, n_sources):
        self.N = N 
        self.n_sources = n_sources
        self.M_ = 1e8

        self.init_bin_problem()
        self.init_coco_problem()

    def init_bin_problem(self):
        in_storage = cp.Variable(self.N+1)
        store_in = cp.Variable(self.N)
        store_out = cp.Variable(self.N)
        buy = cp.Variable((self.n_sources, self.N))
        y = cp.Variable(self.N, boolean=True)

        self.bin_prob_vars = {}
        self.bin_prob_vars['y'] = y
        self.bin_prob_vars['in_storage'] = in_storage
        self.bin_prob_vars['store_in'] = store_in
        self.bin_prob_vars['store_out'] = store_out
        self.bin_prob_vars['buy'] = buy

        initial_storage_MW = cp.Parameter()
        max_discharges = cp.Parameter()
        supply = cp.Parameter((self.n_sources, self.N))
        hour_ahead_forecast = cp.Parameter((self.N))

        self.bin_prob_params = {}
        self.bin_prob_params['max_discharges'] = max_discharges
        self.bin_prob_params['initial_storage_MW'] = initial_storage_MW
        self.bin_prob_params['supply'] = supply
        self.bin_prob_params['hour_ahead_forecast'] = hour_ahead_forecast

        cons = []
        cons += [in_storage >= 0.0]
        cons += [store_in >= 0.0]
        cons += [store_out >= 0.0]
        cons += [buy >= 0.0]

        # Storage initialization constraint
        cons += [in_storage[0] == initial_storage_MW]

        for ii in range(self.N):
            # Available power constraint
            for jj in range(self.n_sources):
                cons += [buy[jj, ii] <= supply[jj, ii]]

            # Storage conservation of energy
            cons += [in_storage[ii+1] == in_storage[ii] + store_in[ii] - store_out[ii]]

            # Sufficiency constraint
            cons += [sum(buy[:, ii]) - store_in[ii] + store_out[ii] >= hour_ahead_forecast[ii]]

            # Store-in constraint
            cons += [store_in[ii] <= sum(buy[:, ii])]

            # Store-out constraint
            cons += [store_out[ii] <= self.M_*y[ii]]

        for ii in range(0, self.N, 3):
            min_idx, max_idx = ii, np.minimum(ii+3, self.N-1)
            cons += [cp.sum(y[min_idx:max_idx]) <= max_discharges]

        total_cost = cp.sum(buy)
        self.bin_prob = cp.Problem(cp.Minimize(total_cost), cons)

    def solve_bin_problem(self, params, solver=cp.GUROBI):
        # Set cvxpy parameters to their values
        for pp in params:
            self.bin_prob_params[pp].value = params[pp]

        prob_success, cost, solve_time, optvals = False, np.Inf, np.Inf, None
        self.bin_prob.solve(solver=solver)

        if self.bin_prob.status in ['optimal', 'optimal_inaccurate'] and self.bin_prob.status not in ['infeasible', 'unbounded']:
            prob_success = True
            cost = self.bin_prob.value
            optvals = [self.bin_prob_vars['y'].value]

        solve_time = self.bin_prob.solver_stats.solve_time

        # Clear any saved params
        for pp in params:
            self.bin_prob_params[pp].value = None

        return prob_success, cost, solve_time, optvals

    def init_coco_problem(self):
        in_storage = cp.Variable(self.N+1)
        store_in = cp.Variable(self.N)
        store_out = cp.Variable(self.N)
        buy = cp.Variable((self.n_sources, self.N))

        self.coco_prob_vars = {}
        self.coco_prob_vars['in_storage'] = in_storage
        self.coco_prob_vars['store_in'] = store_in
        self.coco_prob_vars['store_out'] = store_out
        self.coco_prob_vars['buy'] = buy

        initial_storage_MW = cp.Parameter()
        max_discharges = cp.Parameter()
        supply = cp.Parameter((self.n_sources, self.N))
        hour_ahead_forecast = cp.Parameter(self.N)
        y = cp.Parameter(self.N) 

        self.coco_prob_params = {}
        self.coco_prob_params['max_discharges'] = max_discharges
        self.coco_prob_params['initial_storage_MW'] = initial_storage_MW
        self.coco_prob_params['supply'] = supply
        self.coco_prob_params['hour_ahead_forecast'] = hour_ahead_forecast
        self.coco_prob_params['y'] = y

        cons = []
        cons += [in_storage >= 0.0]
        cons += [store_in >= 0.0]
        cons += [store_out >= 0.0]
        cons += [buy >= 0.0]

        # Storage initialization constraint
        cons += [in_storage[0] == initial_storage_MW]

        for ii in range(self.N):
            # Available power constraint
            for jj in range(self.n_sources):
                cons += [buy[jj, ii] <= supply[jj, ii]]

            # Storage conservation of energy
            cons += [in_storage[ii+1] == in_storage[ii] + store_in[ii] - store_out[ii]]

            # Sufficiency constraint
            cons += [sum(buy[:, ii]) - store_in[ii] + store_out[ii] >= hour_ahead_forecast[ii]]

            # Store-in constraint
            cons += [store_in[ii] <= sum(buy[:, ii])]

            # Store-out constraint
            cons += [store_out[ii] <= self.M_*y[ii]]

        for ii in range(0, self.N, 3):
            min_idx, max_idx = ii, np.minimum(ii+3, self.N-1)
            cons += [cp.sum(y[min_idx:max_idx]) <= max_discharges]

        total_cost = cp.sum(buy)
        self.coco_prob = cp.Problem(cp.Minimize(total_cost), cons)

    def solve_coco_problem(self, params, y_guess, solver=cp.GUROBI):
        # Set cvxpy parameters to their values
        for pp in params:
            self.coco_prob_params[pp].value = params[pp]

        self.coco_prob_params['y'].value = y_guess
        self.coco_prob.solve(solver=solver)

        prob_success, cost, solve_time = False, np.Inf, np.Inf
        if self.coco_prob.status in ['optimal', 'optimal_inaccurate'] and self.coco_prob.status not in ['infeasible', 'unbounded']:
            prob_success = True
            cost = self.coco_prob.value
        solve_time = self.bin_prob.solver_stats.solve_time

        # Clear any saved params
        for pp in self.coco_prob_params:
            self.coco_prob_params[pp].value = None

        return prob_success, cost, solve_time
