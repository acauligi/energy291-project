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
        y = cp.Variable((self.n_sources, self.N), boolean=True)

        self.bin_prob_vars = {}
        self.bin_prob_vars['y'] = y
        self.bin_prob_vars['in_storage'] = in_storage
        self.bin_prob_vars['store_in'] = store_in
        self.bin_prob_vars['store_out'] = store_out
        self.bin_prob_vars['buy'] = buy

        initial_storage_MW = cp.Parameter()
        max_sources = cp.Parameter()
        supply = cp.Parameter((self.n_sources, self.N))
        hour_ahead_forecast = cp.Parameter((self.N))

        self.bin_prob_params = {}
        self.bin_prob_params['initial_storage_MW'] = initial_storage_MW
        self.bin_prob_params['max_sources'] = max_sources
        self.bin_prob_params['supply'] = supply
        self.bin_prob_params['hour_ahead_forecast'] = hour_ahead_forecast

        cons = []

        # Upper and lower bounds
        cons += [in_storage >= 0.0]
        cons += [store_in >= 0.0]
        cons += [store_out >= 0.0]
        cons += [buy >= 0.0]

        # Storage initialization constraint
        cons += [in_storage[0] == initial_storage_MW]

        for ii in range(self.N):
            # Available power constraint
            for jj in range(self.n_sources):
                cons += [buy[jj,ii] <= supply[jj,ii]]

            # Storage conservation of energy
            cons += [in_storage[ii+1] == in_storage[ii] + store_in[ii] - store_out[ii]]

            # Sufficiency constraint
            cons += [sum(buy[:,ii]) - store_in[ii] + store_out[ii] >= hour_ahead_forecast[ii]]

            # Store-in constraint
            cons += [store_in[ii] <= sum(buy[:,ii])]

            for jj in range(self.n_sources):
                cons += [buy[jj,ii] <= self.M_*y[jj,ii]]
            cons += [cp.sum(y[:,ii]) <= max_sources]

        sf = 1e6
        total_cost = cp.sum(buy)
        regularization_factor = 1e-4
        for ii in range(self.n_sources):
            total_cost += regularization_factor*cp.quad_form(buy[ii,:], np.eye(self.N))

        self.bin_prob = cp.Problem(cp.Minimize(total_cost), cons)

    def solve_bin_problem(self, params, solver=cp.GUROBI):
        # Set cvxpy parameters to their values
        for pp in params:
            try:
                self.bin_prob_params[pp].value = params[pp]
            except:
                pdb.set_trace()

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
        max_sources = cp.Parameter()
        supply = cp.Parameter((self.n_sources, self.N))
        hour_ahead_forecast = cp.Parameter((self.N))
        y = cp.Variable((self.n_sources, self.N), boolean=True)

        self.coco_prob_params = {}
        self.coco_prob_params['initial_storage_MW'] = initial_storage_MW
        self.coco_prob_params['max_sources'] = max_sources
        self.coco_prob_params['supply'] = supply
        self.coco_prob_params['hour_ahead_forecast'] = hour_ahead_forecast
        self.coco_prob_params['y'] = y

        cons = []

        # Upper and lower bounds
        cons += [in_storage >= 0.0]
        cons += [store_in >= 0.0]
        cons += [store_out >= 0.0]
        cons += [buy >= 0.0]

        # Storage initialization constraint
        cons += [in_storage[0] == initial_storage_MW]

        for ii in range(self.N):
            # Available power constraint
            for jj in range(self.n_sources):
                cons += [buy[jj,ii] <= supply[jj,ii]]

            # Storage conservation of energy
            cons += [in_storage[ii+1] == in_storage[ii] + store_in[ii] - store_out[ii]]

            # Sufficiency constraint
            cons += [sum(buy[:,ii]) - store_in[ii] + store_out[ii] >= hour_ahead_forecast[ii]]

            # Store-in constraint
            cons += [store_in[ii] <= sum(buy[:,ii])]

            for jj in range(self.n_sources):
                cons += [buy[jj,ii] <= self.M_*y[jj,ii]]
            cons += [cp.sum(y[:,ii]) <= max_sources]

        sf = 1e6
        total_cost = cp.sum(buy)
        regularization_factor = 1e-4
        for ii in range(self.n_sources):
            total_cost += regularization_factor*cp.quad_form(buy[ii,:], np.eye(self.N))

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
