"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class containing the methods for the creation and modification of the solutions stored in the solution archive.
                Algorithm based on the paper: "Ant Colony Optimization for Mixed-Variable Optimization Problems",
                T. Liao, K. Socha, M. Montes de Oca, T. Stuetzle - August 2014
                The implementation is limited to continuous (R) and ordinal (O) variables tailored to application to Mixed Integer Non-Linear Programming (MINLP) Models
==========================================================================================================================================================================="""

import numpy as np

class Solutions:

    def __init__(self, object, run):

        ## extract hyperparameters used in the class
        # number of artificial ants associated to the number of solutions
        self.k = object.nlpDict['hyperparameters']['kNumberArray'].size
        # scaling factor in the weighting function of the solutions ranking
        self.q = object.nlpDict['hyperparameters']['q']
        # number of artificial ants associated to the number of new solutions created
        self.m = object.nlpDict['hyperparameters']['mNumberArray'].size

        # instantiate the number of run to which the solution instance is associate
        self.run = run
        # instantiate the parent class instance
        self.object = object

        # compute the probability distribution based on the solution weights
        self.weight()
        # construct the matrices of the decision variables' domain used in the solutions modification
        self.matrixConstruction()
        # initialize solution archives the arrays of objective function
        self.initializeSA()
        # fill the solution archive with randomly generated solutions
        self.randomSolutions()

    def weight(self):
        """method to compute the probability of choice of each solution in the step of solution modification
        :return: array with discrete probability distribution
        """
        # weight associated to the solution ranking
        omega = self.gaussRankingWeights()
        # discrete probability distribution over the k solutions in archive
        self.probability = omega/omega.sum()

    def gaussRankingWeights(self):
        """
        Method computing the weights of each solution based on its ranking position.
        :input: number of solutions, integer type
        :return: array containing the weight assigned to each solution based on the implemented function
        """
        rank = np.arange(1, self.k + 1)

        a_coeff = 1 / (self.q * self.k * np.sqrt(2 * np.pi))
        b_coeff = -1 / (2 * np.power(self.q, 2) * np.power(self.k, 2))

        return a_coeff * np.exp(b_coeff * np.power(rank - 1, 2))

    def matrixConstruction(self):
        """construct matrices with input data of the decision variables' domain used for the solution modification
        :return: decision variables' domain in matrix format
        """
        self.LB_r = np.dot(np.ones([self.k, 1]), self.object.dictVars['R']['LBArray'])
        self.UB_r = np.dot(np.ones([self.k, 1]), self.object.dictVars['R']['UBArray'])

        self.LB_o = np.dot(np.ones([self.k, 1], dtype=np.int), \
                           self.object.dictVars['O']['LBArray'])
        self.UB_o = np.dot(np.ones([self.k, 1], dtype=np.int), \
                           self.object.dictVars['O']['UBArray'])

    def initializeSA(self):
        """ initialize solution archives and array containing the fitness of the solutions per solution
        :return: matrices and arrays initialized to zero
        """
        self.SA_r = np.zeros([self.k, self.object.dictVars['R']['n']])
        self.SA_o = np.zeros([self.k, self.object.dictVars['O']['n']], dtype=np.int)

        self.f = np.zeros(self.k)

        self.SA_r_new = np.zeros([self.m, self.object.dictVars['R']['n']])
        self.SA_o_new = np.zeros([self.m, self.object.dictVars['O']['n']], dtype=np.int)

        self.f_new = np.zeros(self.m)

    def randomSolutions(self):
        """method to randomly generate new solutions and fill the solution archive.
        :return: matrices of solution archive filled with new solutions
        """

        for j in self.object.nlpDict['hyperparameters']['kNumberArray']:

            for i in self.object.dictVars['R']['idxArray']:
                lowerBoundValue = self.object.dictVars['R']['LBArray'][0,i]
                upperBoundValue = self.object.dictVars['R']['UBArray'][0,i]
                self.SA_r[j, i] = np.random.uniform(lowerBoundValue, upperBoundValue)

            for i in self.object.dictVars['O']['idxArray']:
                lowerBoundValue = self.object.dictVars['O']['LBArray'][0,i]
                upperBoundValue = self.object.dictVars['O']['UBArray'][0,i]
                self.SA_o[j, i] = np.random.randint(lowerBoundValue, upperBoundValue)
