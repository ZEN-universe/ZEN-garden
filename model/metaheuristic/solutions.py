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
import pyomo.environ as pe
import numpy as np
from scipy import stats

class Solutions:

    def __init__(self, object, run):

        ## extract hyperparameters used in the class
        # number of artificial ants associated to the number of solutions
        self.k = object.nlpDict['hyperparameters']['kNumberArray'].size
        # scaling factor in the weighting function of the solutions ranking
        self.q = object.nlpDict['hyperparameters']['q']
        # number of artificial ants associated to the number of new solutions created
        self.m = object.nlpDict['hyperparameters']['mNumberArray'].size
        # hyperparameter defining the standard deviation in pdf for solution construction
        self.xi = object.nlpDict['hyperparameters']['xi']
        # value used as substitute of zero to avoid numerical errors
        self.minValue = object.nlpDict['hyperparameters']['minVal']

        ## parameters from higher level class
        # instantiate the number of run to which the solution instance is associate
        self.run = run
        # instantiate the parent class instance
        self.object = object

        ## initialise set of solutions in each run
        # compute the probability distribution based on the solution weights
        self.weight()
        # construct the matrices of the decision variables' domain used in the solutions modification
        self.matrixConstruction()
        # initialize solution archives and the objective function array
        self.initializeSA()

    def solutionSets(self, step=''):
        """modify values in the solution archive either as random assignment or based on pdf in the solution domain.
        :return: solution archive with new values, array of solution indices
        """

        if step == '':
            # fill the solution archive with randomly generated solutions
            self.randomSolutions()
            solutionsIndices = self.object.nlpDict['hyperparameters']['kNumberArray']
            SA = self.SA

        elif step == 'new':
            # modify the solutions according to pdf
            self.constructSolutions()
            solutionsIndices = self.object.nlpDict['hyperparameters']['mNumberArray']
            SA = self.SA_new

        return solutionsIndices, SA

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
        self.LB = {}
        self.UB = {}
        for type in ['R', 'O']:
            self.LB[type] = np.dot(np.ones([self.k, 1]), self.object.dictVars[type]['LBArray'])
            self.UB[type] = np.dot(np.ones([self.k, 1]), self.object.dictVars[type]['UBArray'])

    def initializeSA(self):
        """ initialize solution archives and array containing the fitness of the solutions per solution
        :return: matrices and arrays initialized to zero
        """
        self.SA = {}
        self.SA['R'] = np.zeros([self.k, self.object.dictVars['R']['n']])
        self.SA['O'] = np.zeros([self.k, self.object.dictVars['O']['n']], dtype=np.int)

        self.f = np.zeros(self.k)

        self.SA_new = {}
        self.SA_new['R'] = np.zeros([self.m, self.object.dictVars['R']['n']])
        self.SA_new['O'] = np.zeros([self.m, self.object.dictVars['O']['n']], dtype=np.int)

        self.f_new = np.zeros(self.m)

    def randomSolutions(self):
        """method to randomly generate new solutions and fill the solution archive.
        :return: matrices of solution archive filled with new solutions
        """

        for j in self.object.nlpDict['hyperparameters']['kNumberArray']:

            for type in ['R', 'O']:

                for i in self.object.dictVars[type]['idxArray']:
                    lowerBoundValue = self.object.dictVars[type]['LBArray'][0,i]
                    upperBoundValue = self.object.dictVars[type]['UBArray'][0,i]
                    if type == 'R':
                        self.SA[type][j, i] = np.random.uniform(lowerBoundValue, upperBoundValue)
                    elif type == 'O':
                        self.SA[type][j, i] = np.random.randint(lowerBoundValue, upperBoundValue + 1)


    def constructSolutions(self):
        """modify the solution archive according to a probability density function defined in the solution space
        :return: the modified instance of the solution archive
        """
        # random selection of m solutions for modification
        solutions = np.random.choice(self.object.nlpDict['hyperparameters']['kNumberArray'], self.m, p=self.probability)

        # create new solutions in the archive
        for type in ['R', 'O']:
            self.SA_new[type][:, :] = self.computeNewValues(self.SA[type], self.LB[type], self.UB[type], solutions)

    def computeNewValues(self, SA, LB, UB, solutions):
        """method to compute the new solutions based on vectorisation.
        :return: the new solution archive matrix
        """

        def get_truncated_normal(mean=0, sd=1, low=0, up=10):
            """from: https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
            """
            return stats.truncnorm((low - mean) / sd, (up - mean) / sd, loc=mean, scale=sd)

        # tensor of three dimensions based on the solution archive matrix
        SA_ts = np.matmul(SA[:, :, None], np.ones([1, SA.shape[0]]))

        # sum of absolute residuals of the single element SA[j,i] with all the values in SA[:,i]
        SAR = abs(SA_ts - SA_ts.T).sum(axis=2)

        SIGMA = self.xi / (self.k - 1) * SAR
        SIGMA[SIGMA < self.minValue] = self.minValue

        # create a new solution archive matrix based on the m selected solutions
        SA_new = get_truncated_normal(mean=SA[solutions, :], sd=SIGMA[solutions, :], low=LB[solutions, :], up=UB[solutions, :]).rvs()

        return SA_new

    def updateObjective(self, instance, solutionIndex, step):
        """ method to assign the objective function associated ot each single solution generated
        :return: scalar value of the objective function
        """

        # get the objective function value from the model instance and assign it to the array of the solution archive
        if step == '':
            self.f[solutionIndex] = pe.value(instance.objective)

        elif step == 'new':
            self.f_new[solutionIndex] = pe.value(instance.objective)

    def rank(self, step):
        """method to rank the solutions according to the value of the objective function.
        :return: solution archive with rows ordered according to the values of the objective function
        """

        SA_temp = {}

        if step == '':
            for type in ['R', 'O']:
                SA_temp[type] = np.zeros([self.k, self.object.dictVars[type]['n']])
            f_temp = np.zeros([self.k])

        elif step == 'new':
            for type in ['R', 'O']:
                SA_temp[type] = np.zeros([self.k + self.m, self.object.dictVars[type]['n']])
            f_temp = np.zeros([self.k + self.m])

            j0 = self.k
            jn = self.k + self.m
            for type in ['R', 'O']:
                SA_temp[type][j0:jn, :] = self.SA_new[type][:, :]
            f_temp[j0:jn] = self.f_new[:]

        j0 = 0
        jn = self.k
        for type in ['R', 'O']:
            SA_temp[type][j0:jn, :] = self.SA[type][:, :]
        f_temp[j0:jn] = self.f[:]

        if self.object.analysis['sense'] == 'minimize':
            argsort = np.argsort(f_temp)
        elif self.analysis['sense'] == 'maximize':
            argsort = np.argsort(-f_temp)

        j0 = 0
        jn = self.k
        for type in ['R', 'O']:
            self.SA[type][:, :] = SA_temp[type][argsort, :][j0:jn, :]
        self.f[:] = f_temp[argsort][j0:jn]