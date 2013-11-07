"""
A few simple graph model classes that fall into the Aldous-Hoover framework.

Scott Linderman
11/7/2013
"""

import numpy as np
from scipy.special import betaln
import matplotlib.pyplot as plt

import string 
import logging
import copy

from sample_utils import *
#from elliptical_slice import *
#from hmc import *

BUG = False
BUG2 = False
BUG3 = False

class AldousHooverNetwork:
    """
    Base class for Aldous-Hoover random graphs
    """
    def __init__(self, A=None):
        """
        Initialize a random Aldous-Hoover random graph
        """
        pass
        
    def pr_A_given_f(self,f1,f2,theta):
        """
        Compute the probability of an edge from a node with 
        features f1 to a node with features f2, given parameters
        theta.
        """
        raise Exception("pr_A_given_f is not implemented!")
    
    def logpr_f(self,f,theta):
        """
        Compute the prior probability of feature f.
        """
        raise Exception("logpr_f is not implemented!")
    
    def logpr_theta(self,theta):
        """
        Compute the prior probability of parameters theta.
        """
        raise Exception("logpr_theta is not implemented!")
    
    def sample_f(self, theta, (n,A,f)=(None,None,None)):
        """
        Sample a set of features. If n,A, and f are given, 
        sample the features of the n-th node from the posterior 
        having observed A and the other features. 
        """
        raise Exception("sample f is not implemented!")
    
    def sample_theta(self, (A,f)=(None,None)):
        """
        Sample the parameters of pr_A_given_f. If A and f
        are given, sample these parameters from the posterior
        distribution.
        """
        raise Exception("sample_theta is not implemented!")
    
    def sample_A(self, f, theta):
        """
        Sample A given features f and parameters theta
        """ 
        N = len(f)
        A = np.zeros((N,N))
        for i in np.arange(N):
            for j in np.arange(N):
                A[i,j] = self.sample_Aij(f[i],f[j],theta)
        
        return A
    
    def sample_Aij(self, fi, fj, theta):
        """
        Sample a single entry in the network
        """
        return np.random.rand() < self.pr_A_given_f(fi,fj, theta)
    
    def logpr(self, A, f, theta):
        """
        Compute the log probability of a network given the 
        node features, the parameters, and the adjacency matrix A.
        """
        lp = 0.0            
        lp += self.logpr_theta(theta)
                
        for fi in f:
            lp += self.logpr_f(fi, theta)
        
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[0]):
                lp += A[i,j]*np.log(self.pr_A_given_f(f[i], f[j], theta)) + \
                      (1-A[i,j])*np.log(1-self.pr_A_given_f(f[i], f[j], theta))        
        return lp
    
class ErdosRenyiNetwork(AldousHooverNetwork):
    """
    Model an Erdos-Renyi random graph
    """
    def __init__(self, rho=None, x=None):
        """ Constructor.
        :param rho    sparsity of the graph (probability of an edge)
        :param (a,b)  parameters of a Beta prior on rho
        
        Either rho or (a,b) must be specified
        """
#        super(ErdosRenyiNetwork,self).__init__()
        if rho is None:
            if x is None:
                raise Exception("Either rho or (a,b) must be specified")
            else:
                (a,b) = x
        else:
            if x is not None:
                raise Exception("Either rho or (a,b) must be specified")
            else:
                a = None
                b = None
                        
        self.rho = rho
        self.a = a
        self.b = b
        
    def pr_A_given_f(self,fi,fj, theta):
        """
        The probability of an edge is simply theta, regardless
        of the "features" fi and fj.
        """
        rho = theta
        return rho
        
    def logpr_f(self,f,theta):
        """
        There are no features for the Erdos-Renyi graph
        """
        return 0.0
    
    def logpr_theta(self,theta):
        """
        This varies depending on whether or not rho is specified
        """
        rho = theta
        if self.rho is not None:
            if self.rho == rho:
                return 0.0
            else:
                return -np.Inf
            
        else:
            return (self.a-1.0)*np.log(rho) + (self.b-1.0)*np.log((1.0-rho)) - \
                   betaln(self.a,self.b)
    
    def sample_f(self, theta, (n,A,f)=(None,None,None)):
        """
        There are no features in the Erdos-Renyi graph model.
        """
        return 1.0
    
    def sample_theta(self, (A,f)=(None,None)):
        """
        Sample the parameters of pr_A_given_f. For the Erdos-Renyi
        graph model, the only parameter is rho.
        """
        if self.rho is not None:
            return self.rho
        elif A is None and f is None:
            return np.random.beta(self.a,self.b)
        else:
            N = A.shape[0]
            nnz_A = np.sum(A)
            a_post = self.a + nnz_A
            b_post = self.b +  N**2 - nnz_A
            return np.random.beta(a_post, b_post)
            
class StochasticBlockModel(AldousHooverNetwork):
    """
    Model an Erdos-Renyi random graph
    """
    def __init__(self, R, b0, b1, alpha0):
        """ Constructor.
        :param R     Number of blocks
        :param b0    prior probability of edge
        :param b1    prior probability of no edge
        :param alpha prior probability of block membership
        
        Either rho or (a,b) must be specified
        """
#        super(ErdosRenyiNetwork,self).__init__()
        self.R = R
        self.b0 = b0
        self.b1 = b1
        self.alpha0 = alpha0
                
    def pr_A_given_f(self,fi,fj, theta):
        """
        The probability of an edge is beta distributed given 
        the blocks fi and fj
        """
        zi = fi
        zj = fj
        (B,pi) = theta
        return B[zi,zj]
        
    def logpr_f(self,f,theta):
        """
        The features of a stochastic block model are the nodes'
        block affiliations
        """
        (B,pi) = theta
        fint = np.array(f).astype(np.int)
        lp = 0.0
        lp += np.sum(np.log(pi[fint])) 
        return lp
    
    def logpr_theta(self,theta):
        """
        This varies depending on whether or not rho is specified
        """
        (B,pi) = theta
        lp = 0.0
        
        # Add prior on B
        for ri in np.arange(self.R):
            for rj in np.arange(self.R):
                lp += (self.b1-1.0)*np.log(B[ri,rj]) + (self.b0-1.0)*np.log((1.0-B[ri,rj])) - \
                       betaln(self.b1,self.b0)
        
        # Add prior on pi
        for ri in np.arange(self.R):
            lp += np.sum((self.alpha0-1.0)*np.log(pi))
            
        return lp
    
    def sample_f(self, theta, (n,A,f)=(None,None,None)):
        """
        Sample new block assignments given the parameters.
        """
        (B,pi) = theta
        if n is None and A is None and f is None:
            # Sample the prior
            zn = discrete_sample(pi)
        else:
            # Sample the conditional distribution on f[n]
            zn = self.naive_sample_f(theta, n, A, f)
#            zn = self.collapsed_sample_f(theta, n, A, f)
    
        return zn
    
    def naive_sample_f(self, theta, n, A, f):
        """
        Naively Gibbs sample z given B and pi
        """
        (B,pi) = theta
        A = A.astype(np.bool)
        zother = np.array(f).astype(np.int)
        
        # Compute the posterior distribution over blocks
        ln_pi_post = np.log(pi)
        
        if BUG3:
            rrange = np.arange(1,self.R)
        else:
            rrange = np.arange(self.R)        
        for r in rrange:
            zother[n] = r
            # Block IDs of nodes we connect to 
            o1 = A[n,:]
            if np.any(A[n,:]):
                ln_pi_post[r] += np.sum(np.log(B[np.ix_([r],zother[o1])]))
            
            # Block IDs of nodes we don't connect to
            o2 = np.logical_not(A[n,:])
            if np.any(o2):
                ln_pi_post[r] += np.sum(np.log(1-B[np.ix_([r],zother[o2])]))
            
            # Block IDs of nodes that connect to us
            i1 = A[:,n]
            if np.any(i1):
                ln_pi_post[r] += np.sum(np.log(B[np.ix_(zother[i1],[r])]))

            # Block IDs of nodes that do not connect to us
            i2 = np.logical_not(A[:,n])
            if np.any(i2):
                ln_pi_post[r] += np.sum(np.log(1-B[np.ix_(zother[i2],[r])]))
            
        if BUG2:
            # Introduce numerical error by not using log sum exp sampling
            # This, unfortunately, doesn't seem to make much difference.
            # Perhaps it would if the network were larger and numerical 
            # issues were more significant
            pi_post = np.exp(ln_pi_post)/np.sum(np.exp(ln_pi_post))
            zn = discrete_sample(pi_post)
        else:
            zn = log_sum_exp_sample(ln_pi_post)
        
        return zn
    
    def collapsed_sample_f(self, theta,n,A,f):
        """
        Use a collapsed Gibbs sampler to update the block assignments 
        by integrating out the block-to-block connection probabilities B.
        Since this is a Beta-Bernoulli model the posterior can be computed
        in closed form and the integral can be computed analytically.
        """
        (B,pi) = theta
        A = A.astype(np.bool)
        zother = np.array(f).astype(np.int)
        
        # P(A|z) \propto 
        #    \prod_{r1}\prod_{r2} Beta(m(r1,r2)+b1,\hat{m}(r1,r2)+b0) /
        #                           Beta(b1,b0)
        # 
        # Switching z changes the product over r1 and the product over r2
        
        # Compute the posterior distribution over blocks
        
        # TODO: This literal translation of the log prob is O(R^3)
        # But it can almost certainly be sped up to O(R^2)
        ln_pi_post = np.log(pi)
        for r in np.arange(self.R):
            zother[n] = r
            for r1 in np.arange(self.R):
                for r2 in np.arange(self.R):
                    # Look at outgoing edges under z[n] = r
                    Ar1r2 = A[np.ix_(zother==r1,zother==r2)]
                    mr1r2 = np.sum(Ar1r2)
                    hat_mr1r2 = Ar1r2.size - mr1r2
                    
                    ln_pi_post[r] += betaln(mr1r2+self.b1, hat_mr1r2+self.b0) - \
                                     betaln(self.b1,self.b0)
                                
            zn = log_sum_exp_sample(ln_pi_post)
        
        return zn
        
    def sample_theta(self, (A,f)=(None,None)):
        """
        Sample the parameters of pr_A_given_f. For the Erdos-Renyi
        graph model, the only parameter is rho.
        """
        if A is None and f is None:
            # Sample B and pi from the prior
            B = np.random.beta(self.b1,self.b0,
                               (self.R,self.R))
            pi = np.random.dirichlet(self.alpha0)
        else:
            # Sample pi from its Dirichlet posterior
            z = np.array(f).astype(np.int)
            alpha_post = np.zeros((self.R,))
            for r in np.arange(self.R):
                alpha_post[r] = self.alpha0[r] + np.sum(z==r)
            pi = np.random.dirichlet(alpha_post)
            
            # Sample B from its Beta posterior
            B = np.zeros((self.R,self.R), dtype=np.float32)
            for r1 in np.arange(self.R):
                for r2 in np.arange(self.R):
                    b0post = self.b0
                    b1post = self.b1
                    
                    Ar1r2 = A[np.ix_(z==r1, z==r2)]
                    if np.size(Ar1r2) > 0:
                        b0post += np.sum(1-Ar1r2)
                        b1post += np.sum(Ar1r2)
                    
                    if BUG:
                        # Introduce an artificial bug that should be caught 
                        # by Geweke validation. Swap the order of the input
                        # parameters
                        B[r1,r2] = np.random.beta(b0post, b1post)
                    else:
                        B[r1,r2] = np.random.beta(b1post, b0post)
        
        return (B,pi)
                
# Define helper functions to sample a random graph
def sample_network(model, N):
    """
    Sample a new network with N nodes from the given model.
    """
    # First sample theta, the parameters of the base measure
    theta = model.sample_theta()
    
    # Then sample features for each node
    f = []
    for n in np.arange(N):
        f.append(model.sample_f(theta))
                
    # Finally sample the network itself
    A = model.sample_A(f,theta)
    
    return (A,f,theta)

def fit_network(A, model, x0=None, N_iter=1000, callback=None, pause=False):
    """
    Fit the parameters of the network model using MCMC.
    """
    N = A.shape[0]
    
    # If the initial features are not specified, start with a 
    # draw from the prior.
    if x0 is None:
        theta0 = model.sample_theta()
        
        f0 = []
        for n in np.arange(N):
            f0.append(model.sample_f(theta0))
            
    else:
        (f0,theta0) = x0       

    print "Starting Gibbs sampler"    
    f = copy.deepcopy(f0)
    theta = copy.deepcopy(theta0)
    
    lp_trace = np.zeros(N_iter)
    f_trace = []
    theta_trace = []
    for iter in np.arange(N_iter):
        lp = model.logpr(A,f,theta)
        lp_trace[iter] = lp
        
        print "Iteration %d. \tlog pr: %f" % (iter, lp_trace[iter])
        
        # Sample the model parameters theta
        theta = model.sample_theta((A,f))
        
        # Sample features f
        for n in np.arange(N):
            f[n] = model.sample_f(theta, (n,A,f))
        
        # If the user supplied a callback, call it now
        if callback is not None:
            callback(f, theta)
        
        f_trace.append(f)
        theta_trace.append(theta)
        
        if pause:
            raw_input("Press enter to continue.")
    
    return (f_trace, theta_trace, lp_trace)


def geweke_test(N, model, N_iter=1000, callback=None, pause=False):
    """
    Fit the parameters of the network model using MCMC.
    """    
    # If the initial features are not specified, start with a 
    # draw from the prior.
    theta0 = model.sample_theta()
    
    f0 = []
    for n in np.arange(N):
        f0.append(model.sample_f(theta0))
          
    A0 = model.sample_A(f0, theta0)

    print "Starting Gibbs sampler"    
    f = copy.deepcopy(f0)
    theta = copy.deepcopy(theta0)
    A = np.copy(A0)
    
    lp_trace = np.zeros(N_iter)
    f_trace = []
    theta_trace = []
    A_trace = []
    for iter in np.arange(N_iter):
        lp = model.logpr(A,f,theta)
        lp_trace[iter] = lp
        
        print "Iteration %d. \tlog pr: %f" % (iter, lp_trace[iter])
        
        # Sample the model parameters theta
        theta = model.sample_theta((A,f))
        
        # Sample features f
        for n in np.arange(N):
            f[n] = model.sample_f(theta, (n,A,f))
        
        # Sample a new graph A given the updated features
        A = model.sample_A(f, theta)
        
        # If the user supplied a callback, call it now
        if callback is not None:
            callback(A, f, theta)
        
        # Save a copy of the data
        theta_trace.append(copy.deepcopy(theta))
        f_trace.append(copy.deepcopy(f))
        A_trace.append(A.copy())
        
        if pause:
            raw_input("Press enter to continue.")
    
    return (f_trace, theta_trace, lp_trace)

#class LatentDistanceModel(LogisticGraphModelExtension):
#    """
#    Prior for a latent distance model of connectivity. Each process is given 
#    a location in some latent space, and the probability of connectivity is 
#    exponentially decreasing with distance between two processes. 
#    """
#    def __init__(self, baseModel, configFile):
#        super(LatentDistanceModel,self).__init__(baseModel, configFile)
#        
#        self.parseConfigurationFile(configFile)
#        pprintDict(self.params, "Graph Model Params")
#        
#        self.params["registered"] = False 
#        
#    def parseConfigurationFile(self, configFile):
#        
#        # Set defaults
#        defaultParams = {}
#        defaultParams["thin"] = 50
#        
#        # Parse config file
#        cfgParser = ConfigParser(defaultParams)
#        cfgParser.read(configFile)
#        
#        distparams = {}
#        distparams["location_name"] = cfgParser.get("graph_model", "location")
#        distparams["mu_theta"] = cfgParser.getfloat("graph_model", "mu_theta")
#        distparams["sig_theta"] = cfgParser.getfloat("graph_model", "sig_theta")
#        distparams["thin"] = cfgParser.getint("graph_model", "thin")
#        
#        # Combine the two param dicts
#        self.params.update(distparams)
#        
#    def initializeModelParamsFromPrior(self):
#        self.register_providers()
#        
#        self.initializeGpuMemory()
#        
#        K = self.modelParams["proc_id_model","K"]
#        
#        # Initialize tau with draw from exponential prior
#        self.modelParams["graph_model", "tau"] = np.exp(np.random.normal(self.params["mu_theta"],
#                                                                         self.params["sig_theta"]))
#        
#        # Initialize the pairwise distance matrix
#        self.computeDistanceMatrix()
#        
#        # Override the base model's default adjacency matrix with a draw from the prior
#        self.modelParams["graph_model","A"] = self.sampleGraphFromPrior()
#        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
#        
#        self.iter = 0
#    
#    def initializeModelParamsFromDict(self, paramsDB):
#        self.register_providers()
#        self.initializeGpuMemory()
#        self.modelParams["graph_model","A"] = paramsDB["graph_model","A"]
#        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
#        
#        if ("graph_model","tau") in paramsDB:
#            self.modelParams["graph_model", "tau"] = np.float(paramsDB["graph_model","tau"])
#        else:
#            # Initialize tau with draw from exponential prior
#            self.modelParams["graph_model", "tau"] = np.exp(np.random.normal(self.params["mu_theta"],
#                                                                             self.params["sig_theta"]))
#        
#        self.computeDistanceMatrix()
#        
#    def register_providers(self):
#        """
#        Register the cluster and location providers.
#        """
#        if self.params["registered"]:
#            return
#        
#        # Now find the correct location model
#        location = None
#        location_name = self.params["location_name"]
#        location_list = self.base.extensions["location_model"]
#        location_list = location_list if isinstance(location_list, type([])) else [location_list]
#        for location_model in location_list:
#            if location_model.name == location_name:
#                # Found the location model!
#                location = location_model
#        if location is None:
#            raise Exception("Failed to find location model '%s' in extensions!" % location_name)
#            
#        self.params["location"] = location
#        
#        # Add the location callback
#        self.params["location"].register_consumer(self.compute_log_lkhd_new_location)
#        
#        self.params["registered"] = True
#    
#    
#    def computeDistanceMatrix(self):
#        """
#        compute the pairwise distances between each process
#        """
#        K = self.modelParams["proc_id_model","K"]
#        L = self.modelParams[self.params["location"].name, "L"]
#        
#        dist = np.zeros((K,K))
#        for i in np.arange(K):
#            for j in np.arange(i+1,K):
#                d_ij = np.linalg.norm(L[i,:]-L[j,:], 2)
#                dist[i,j] = d_ij
#                dist[j,i] = d_ij
#                
#        self.modelParams["graph_model", "dist"] = dist
#    
#    def getConditionalEdgePr(self, ki, kj):
#        """
#        Return the conditional probability of edge A[ki,kj] given the model
#        parameters.  
#        """
#        if not self.params["allow_self_excitation"] and ki==kj:
#            return 0.0
#        
#        else:
#            return np.exp(-1/self.modelParams["graph_model","tau"]*self.modelParams["graph_model","dist"][ki,kj])
#    
#    def sampleModelParameters(self):
#        """
#        Sample process locations and edges in adjacency matrix
#        """
#        self.computeDistanceMatrix()
#        if np.mod(self.iter, self.params["thin"]) == 0:
#            self.sampleTau()
#        
#        # Sample a new graph on every iteration
#        self.sampleA()
#        
#        self.iter += 1
#    
#    def sampleTau(self):
#        """
#        Sample tau using Hybrid Monte Carlo. The log likelihood is a function of the 
#        current graph and the distances between connected and disconnected nodes.
#        """
#        # Set HMC params
#        epsilon = 0.001
#        n_steps = 10
#        
#        # By convention hmc minimizes the negative log likelihood,
#        # so negate the logprob and gradient calculations
#        theta_new = hmc(lambda t: -1.0*self.computeLogProbTau(t), 
#                        lambda t: -1.0*self.computeGradLogProbTau(t), 
#                        epsilon,
#                        n_steps,
#                        np.log(self.modelParams["graph_model", "tau"]))
#        
#        tau_new = np.exp(theta_new)
#        
#        self.modelParams["graph_model","tau"] = tau_new
#    
#    def computeLogProbTau(self, theta):
#        """
#        Compute the log likelihood of the current graph given theta = log(tau)
#        """
#        tau = np.exp(theta)
#        
#        K = self.modelParams["proc_id_model", "K"]
#        
#        # Get the distances between connected neurons and bw disconnected neurons
#        # Ignore the diagonal since the distances are equal to zero
#        dist_conn = self.modelParams["graph_model", "dist"][self.modelParams["graph_model", "A"]]
#        dist_conn = dist_conn[dist_conn>0]
#        N_conn = np.size(dist_conn)
#        dist_noconn = self.modelParams["graph_model", "dist"][np.bitwise_not(self.modelParams["graph_model", "A"])]
#        dist_noconn = dist_noconn[dist_noconn>0]
#        N_noconn = np.size(dist_noconn)
#        
#        # Compute the logprob
#        lpr = 0.0
#        if N_conn > 0:
#            lpr += -1.0*np.sum(dist_conn)/tau
#        if N_noconn > 0:
#            lpr += np.sum(np.log(1-np.exp(-dist_noconn/tau))) 
#        
#        # Contribution from prior
#        lpr += -0.5*(theta - self.params["mu_theta"])**2/self.params["sig_theta"]**2
#        
#        return lpr
#    
#    def computeGradLogProbTau(self, theta):
#        """
#        Compute the gradient of the log likelihood wrt theta = log(tau)
#        """
#        tau = np.exp(theta)
#        K = self.modelParams["proc_id_model", "K"]
#        
#        # Get the distances between connected neurons and bw disconnected neurons
#        dist_conn = self.modelParams["graph_model", "dist"][self.modelParams["graph_model", "A"]]
#        dist_conn = dist_conn[dist_conn>0]
##        N_conn = np.size(dist_conn)
#        dist_noconn = self.modelParams["graph_model", "dist"][np.bitwise_not(self.modelParams["graph_model", "A"])]
#        dist_noconn = dist_noconn[dist_noconn>0]
##        N_noconn = K**2- N_conn
#        
#        grad_lpr = 0.0
#        grad_lpr += np.sum(dist_conn)/(tau**2) 
#        
#        try:
#            grad_lpr += np.sum(-dist_noconn/(tau**2)*np.exp(-dist_noconn/tau)/(1-np.exp(-dist_noconn/tau)))
#        except Exception as e:
#            # Catch FloatingPointErrors
#            log.error("Caught FloatingPointError (underflow?) in GradLogProbTau")
#            log.info(dist_noconn)
#            log.info(tau)
#            
#        # The above derivative is with respect to tau. Multiply by dtau/dtheta
#        grad_lpr *= np.exp(theta)
#        
#        # Add the gradient of the prior over theta
#        grad_lpr += -(theta-self.params["mu_theta"])/(self.params["sig_theta"]**2)
#        
#        return grad_lpr
#        
#    def compute_log_lkhd_new_location(self, k, Lk):
#        """
#        Compute the log likelihood of A given X[k,:] = x 
#        This affects edges into and out of process k
#        """
#        A = self.modelParams["graph_model", "A"]
#        K = self.modelParams["proc_id_model","K"]
#        L = self.modelParams[self.params["location"].name, "L"]
#        tau = self.modelParams["graph_model","tau"]
#        
#        # Compute the updated distance vector from k to other nodes
#        dist_k = np.zeros((K,))
#        for j in np.arange(K):
#            if j != k:
#                dist_k[j] = np.linalg.norm(Lk-L[j,:], 2)
#        
#        # Compute the log likelihood
#        try:
#            ll = 0
#            for j in np.arange(K):
#                if j != k:
#                    ll += (A[j,k]+A[k,j])*(-1/tau*dist_k[j])
#                    
#                    if dist_k[j] == 0:
#                        # If the distance is zero then there must be an edge
#                        if not A[j,k] or not A[k,j]:
#                            ll = -np.Inf
#                    else:   
#                        ll += (2-A[j,k]-A[k,j])*np.log(1-np.exp(-1/tau*dist_k[j]))
#                    
#        except Exception as e:
#            log.info("compute_lkhd_(%d,%s)", k, str(Lk))
#            log.info("L")
#            log.info(L)
#            log.info("dist")     
#            log.info(dist_k)
#            log.info("tau: %f", tau)
#            raise e
#        return ll
#    
#    def sampleGraphFromPrior(self):
#        """
#        Sample a graph from the prior, assuming model params have been set.
#        """
#        K = self.modelParams["proc_id_model","K"]
#        dist = self.modelParams["graph_model","dist"]
#        tau = self.modelParams["graph_model","tau"]
#        
#        A = np.random.rand(K, K) < np.exp(-1/tau*dist)
#        return A
#    
#    def registerStatManager(self, statManager):
#        """
#        Register callbacks with the given StatManager
#        """
#        super(LatentDistanceModel,self).registerStatManager(statManager)
#        
#        K = self.modelParams["proc_id_model","K"]
#        
#        statManager.registerSampleCallback("A_tau", 
#                                           lambda: self.modelParams["graph_model","tau"],
#                                           (1,),
#                                           np.float32)
#        