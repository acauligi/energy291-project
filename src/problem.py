import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxpy as cp
import gurobipy

class Problem():
    def __init__(self, N, n_sources):
        self.N = N 
        self.n_sources = n_sources
        self.M_ = 1e8

    def init_bin_problem(self):
        in_storage = cp.Variable(self.N+1)
        store_in = cp.Variable(self.N)
        store_out = cp.Variable(N)
        buy = cp.Variable((self.n_sources,self.N))
        y = cp.Variable(self.N, boolean=True)

        self.bin_prob_variables = {}
        self.bin_prob_variables['in_storage'] = in_storage
        self.bin_prob_variables['store_in'] = store_in
        self.bin_prob_variables['store_out'] = store_out
        self.bin_prob_variables['buy'] = buy
        self.bin_prob_variables['y'] = y

        initial_storage_MW = cp.Parameter(1)
        max_discharges = cp.Parameter(1)
        supply = cp.Parameter(self.N)
        hour_ahead_forecast = cp.Parameter(self.N)

        self.bin_prob_parameters = {}
        self.bin_prob_parameters['max_discharges'] = max_discharges
        self.bin_prob_parameters['initial_storage_MW'] = initial_storage_MW
        self.bin_prob_parameters['supply'] = supply
        self.bin_prob_parameters['hour_ahead_forecast'] = hour_ahead_forecast

        cons = []
        cons += [in_storage >= 0.0]
        cons += [store_in >= 0.0]
        cons += [store_out >= 0.0]
        cons += [buy >= 0.0]

        # Storage initialization constraint
        cons += [in_storage[0] == initial_storage_MW]

        for ii in range(N):
            # Available power constraint
            for jj in range(n_sources):
                cons += [buy[jj,ii] <= np.maximum(supply[jj,ii], 0.0)]

            # Storage conservation of energy
            cons += [in_storage[ii+1] == in_storage[ii] + store_in[ii] - store_out[ii]]

            # Sufficiency constraint
            cons += [sum(buy[:,ii]) - store_in[ii] + store_out[ii] >= hour_ahead_forecast[ii]]

            # Store-in constraint
            cons += [store_in[ii] <= sum(buy[:,ii])]

            # Store-out constraint
            cons += [store_out[ii] <= self.M_*y[ii]]

        for ii in range(0,N,3):
            min_idx, max_idx = ii, np.minimum(ii+3,N-1)
            cons += [cp.sum(y[min_idx:max_idx]) <= max_discharges]

        total_cost = cp.sum(buy)
        self.bin_prob = cp.Problem(cp.Minimize(total_cost), cons)

    def init_coco_problem(self):
        in_storage = cp.Variable(self.N+1)
        store_in = cp.Variable(self.N)
        store_out = cp.Variable(N)
        buy = cp.Variable((self.n_sources,self.N))

        self.bin_prob_variables = {}
        self.bin_prob_variables['in_storage'] = in_storage
        self.bin_prob_variables['store_in'] = store_in
        self.bin_prob_variables['store_out'] = store_out
        self.bin_prob_variables['buy'] = buy

        initial_storage_MW = cp.Parameter(1)
        max_discharges = cp.Parameter(1)
        supply = cp.Parameter(self.N)
        hour_ahead_forecast = cp.Parameter(self.N)
        y = cp.Parameter(self.N) 

        self.bin_prob_parameters = {}
        self.bin_prob_parameters['max_discharges'] = max_discharges
        self.bin_prob_parameters['initial_storage_MW'] = initial_storage_MW
        self.bin_prob_parameters['supply'] = supply
        self.bin_prob_parameters['hour_ahead_forecast'] = hour_ahead_forecast
        self.bin_prob_parameters['y'] = y

        cons = []
        cons += [in_storage >= 0.0]
        cons += [store_in >= 0.0]
        cons += [store_out >= 0.0]
        cons += [buy >= 0.0]

        # Storage initialization constraint
        cons += [in_storage[0] == initial_storage_MW]

        for ii in range(N):
            # Available power constraint
            for jj in range(n_sources):
                cons += [buy[jj,ii] <= np.maximum(supply[jj,ii], 0.0)]

            # Storage conservation of energy
            cons += [in_storage[ii+1] == in_storage[ii] + store_in[ii] - store_out[ii]]

            # Sufficiency constraint
            cons += [sum(buy[:,ii]) - store_in[ii] + store_out[ii] >= hour_ahead_forecast[ii]]

            # Store-in constraint
            cons += [store_in[ii] <= sum(buy[:,ii])]

            # Store-out constraint
            cons += [store_out[ii] <= self.M_*y[ii]]

        for ii in range(0,N,3):
            min_idx, max_idx = ii, np.minimum(ii+3,N-1)
            cons += [cp.sum(y[min_idx:max_idx]) <= max_discharges]

        total_cost = cp.sum(buy)
        self.bin_prob = cp.Problem(cp.Minimize(total_cost), cons)
