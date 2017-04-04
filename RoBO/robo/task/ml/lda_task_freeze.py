import numpy as np
import time
import cPickle, string, getopt, sys, random, re, pprint
import urllib2, threading
import numpy as n
from scipy.special import gammaln, psi
import os
from itertools import izip
from robo.task.base_task import BaseTask

"""
Adapted from https://github.com/blei-lab/onlineldavb
"""

class LDA(BaseTask):

    def __init__(self, train=None, train_targets=None,
                 valid=None, valid_targets=None,
                 test=None, test_targets=None,
                 n_classes=None, num_epochs=500,
                 save=False, file_name=None):
        '''

        Parameters
        ----------
        train : (N, D) numpy array
            Training matrix where N are the number of data points
            and D are the number of features
        train_targets : (N) numpy array
            Labels for the training data
        valid : (K, D) numpy array
            Validation data
        valid_targets : (K) numpy array
            Validation labels
        test : (L, D) numpy array
            Test data
        test_targets : (L) numpy array
            Test labels
        n_classes: int
            Number of classes in the dataset
        '''
        self.X_train = train
        self.y_train = train_targets
        self.X_val = valid
        self.y_val = valid_targets
        self.X_test = test
        self.y_test = test_targets
        self.num_epochs = num_epochs
        self.save = save
        self.file_name = file_name
        self.base_name = "lda"
        # 1 Dim Number of topics: 2 to 100
        # 2 Dim Dirichlet distribution prior base measure alpha
        # 3 Dim Dirichlet distribution prior base measure eta
        # 4 Dim Learning rate
        # 5 Dim Weight Decay
        X_lower = np.array([2, 0, 0., 0.1, 1e-5])
        X_upper = np.array([100., 2., 2., 1., 1.])

        super(LDA, self).__init__(X_lower, X_upper)




    def set_weights(self, old_file_name):
        file_name = old_file_name + '.pkl'
        with open(file_name, 'rb') as f:
            param_values = cPickle.load(f)
        
        # self.olda._rhot = param_values[0]
        # self.olda._lambda = param_values[1]
        # self.olda._Elogbeta = param_values[2]
        # self.olda._expElogbeta = param_values[3]
        # self.olda._updatect = param_values[4]
        self._rhot = param_values[0]
        self._lambda = param_values[1]
        self._Elogbeta = param_values[2]
        self._expElogbeta = param_values[3]
        self._updatect = param_values[4]

    def set_epochs(self, n_epochs):
        self.num_epochs = n_epochs

    def set_save_modus(self, is_old=True, file_old=None, file_new=None):
        self.save_old = is_old
        self.save = True
        if self.save_old:
            self.file_name = file_old
        else:
            self.file_name = file_new


    def objective_function(self, x):
        # 1 Dim Number of topics: 2 to 100
        # 2 Dim Dirichlet distribution prior base measure alpha
        # 3 Dim Dirichlet distribution prior base measure eta
        # 4 Dim Learning rate
        # 5 Dim Weight Decay        
        topics_number = np.float32(np.exp(x[0, 0]))
        dirichlet_alpha= np.float32(x[0, 1])
        dirichlet_eta = np.int32(x[0, 2])
        weight_decay = np.int32(x[0, 3])
        l1_reg = np.int32(x[0, 4])
        best_perplexity = np.inf

        val_perplexities = []

        """
        Downloads and analyzes a bunch of random Wikipedia articles using
        online VB for LDA.
        """

        # The number of documents to analyze each iteration
        batchsize = 64
        # The total number of documents in Wikipedia
        #D = 250000
        #D = 1000
        D = 3.3e6
        # The number of topics
        K = topics_number

        # How many documents to look at
        #documentstoanalyze = int(D/batchsize)
        documentstoanalyze = 101
        
        # Our vocabulary
        #vocab = file('./dictnostops.txt').readlines()
        vocab = file('dictnostops.txt').readlines()
        W = len(vocab)

        #print 'W: ', W

        # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7

        # self.olda = onlineldavb.OnlineLDA(vocab, K, D, alpha=dirichlet_alpha, eta=dirichlet_eta, tau0=1024., kappa=weight_decay, save=self.save)
        self.olda = OnlineLDA(vocab, K, D, alpha=dirichlet_alpha, eta=dirichlet_eta, tau0=1024., kappa=weight_decay, save=self.save)

        if self.save:
            self.olda.set_file_name(self.file_name)
            self.olda._rhot = self._rhot
            self.olda._lambda = self._lambda
            self.olda._Elogbeta = self._Elogbeta
            self.olda._expElogbeta = self._expElogbeta
            self.olda._updatect = self._updatect

        #print 'documentstoanalyze: ', documentstoanalyze
        # Run until we've seen D documents.

        for iteration in range(0, documentstoanalyze):
            # Download some articles
            #print 'batchsize: ', batchsize
            pre = \
                wikirandom.get_random_wikipedia_articles(batchsize)

            #print '##########################################################'
            #print pre
            #print '##########################################################'

            (docset, articlenames) = pre

            #if self.save:
            #    self.olda.set_file_name(self.file_name)
            # Give them to online LDA
            (gamma, bound) = self.olda.update_lambda_docs(docset)
            # Compute an estimate of held-out perplexity
            # (wordids, wordcts) = onlineldavb.parse_doc_list(docset, self.olda._vocab)
            (wordids, wordcts) = parse_doc_list(docset, self.olda._vocab)

            perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
            perplexity = numpy.exp(-perwordbound)
            val_perplexities.append(perplexity)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity

            #print 'bound: ', bound
            #print 'docset: ', len(docset)
            #print 'perwordbound: ', perwordbound
            print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
                (iteration, self.olda._rhot, perplexity)

            # Save lambda, the parameters to the variational distributions
            # over topics, and gamma, the parameters to the variational
            # distributions over topic weights for the articles analyzed in
            # the last iteration.
            if (iteration % 10 == 0):
                numpy.savetxt('lambda-%d.dat' % iteration, self.olda._lambda)
                numpy.savetxt('gamma-%d.dat' % iteration, gamma)

            return np.array([[best_perplexity]]), val_perplexities
        
    def objective_function_test(self, x):        
        self.objective_function(x)
        return self.test_error


def get_random_wikipedia_article():
    """
    Downloads a randomly selected Wikipedia article (via
    http://en.wikipedia.org/wiki/Special:Random) and strips out (most
    of) the formatting, links, etc. 

    This function is a bit simpler and less robust than the code that
    was used for the experiments in "Online VB for LDA."
    """
    failed = True
    while failed:
        articletitle = None
        failed = False
        try:
            req = urllib2.Request('http://en.wikipedia.org/wiki/Special:Random',
                                  None, { 'User-Agent' : 'x'})
            f = urllib2.urlopen(req)
            while not articletitle:
                line = f.readline()
                result = re.search(r'title="Edit this page" href="/w/index.php\?title=(.*)\&amp;action=edit"/\>', line)
                if (result):
                    articletitle = result.group(1)
                    break
                elif (len(line) < 1):
                    sys.exit(1)

            req = urllib2.Request('http://en.wikipedia.org/w/index.php?title=Special:Export/%s&action=submit' \
                                      % (articletitle),
                                  None, { 'User-Agent' : 'x'})
            f = urllib2.urlopen(req)
            all = f.read()
        except (urllib2.HTTPError, urllib2.URLError):
            print 'oops. there was a failure downloading %s. retrying...' \
                % articletitle
            failed = True
            continue
        print 'downloaded %s. parsing...' % articletitle

        try:
            all = re.search(r'<text.*?>(.*)</text', all, flags=re.DOTALL).group(1)
            all = re.sub(r'\n', ' ', all)
            all = re.sub(r'\{\{.*?\}\}', r'', all)
            all = re.sub(r'\[\[Category:.*', '', all)
            all = re.sub(r'==\s*[Ss]ource\s*==.*', '', all)
            all = re.sub(r'==\s*[Rr]eferences\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks and [Rr]eferences==\s*', '', all)
            all = re.sub(r'==\s*[Ss]ee [Aa]lso\s*==.*', '', all)
            all = re.sub(r'http://[^\s]*', '', all)
            all = re.sub(r'\[\[Image:.*?\]\]', '', all)
            all = re.sub(r'Image:.*?\|', '', all)
            all = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', all)
            all = re.sub(r'\&lt;.*?&gt;', '', all)
        except:
            # Something went wrong, try again. (This is bad coding practice.)
            print 'oops. there was a failure parsing %s. retrying...' \
                % articletitle
            failed = True
            continue

    return(all, articletitle)

class WikiThread(threading.Thread):
    articles = list()
    articlenames = list()
    lock = threading.Lock()

    def run(self):
        (article, articlename) = get_random_wikipedia_article()
        WikiThread.lock.acquire()
        WikiThread.articles.append(article)
        WikiThread.articlenames.append(articlename)
        WikiThread.lock.release()

def get_random_wikipedia_articles(n):
    """
    Downloads n articles in parallel from Wikipedia and returns lists
    of their names and contents. Much faster than calling
    get_random_wikipedia_article() serially.
    """
    maxthreads = 8
    WikiThread.articles = list()
    WikiThread.articlenames = list()
    wtlist = list()
    for i in range(0, n, maxthreads):
        print 'downloaded %d/%d articles...' % (i, n)
        for j in range(i, min(i+maxthreads, n)):
            wtlist.append(WikiThread())
            wtlist[len(wtlist)-1].start()
        for j in range(i, min(i+maxthreads, n)):
            wtlist[j].join()
    return (WikiThread.articles, WikiThread.articlenames)


"""
Online LDA class
"""

# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments: 
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists. 

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())

    return((wordids, wordcts))

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, tau0, kappa, save):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict()
        for word in vocab:
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0
        self.save = save

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            print sum(wordcts[d])
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                print gammad[:, n.newaxis]
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return((gamma, sstats))

    def do_e_step_docs(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = parse_doc_list(docs, self._vocab)

        return self.do_e_step(wordids, wordcts)
    
#         batchD = len(docs)

#         # Initialize the variational distribution q(theta|gamma) for
#         # the mini-batch
#         gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
#         Elogtheta = dirichlet_expectation(gamma)
#         expElogtheta = n.exp(Elogtheta)

#         sstats = n.zeros(self._lambda.shape)
#         # Now, for each document d update that document's gamma and phi
#         it = 0
#         meanchange = 0
#         for d in range(0, batchD):
#             # These are mostly just shorthand (but might help cache locality)
#             ids = wordids[d]
#             cts = wordcts[d]
#             gammad = gamma[d, :]
#             Elogthetad = Elogtheta[d, :]
#             expElogthetad = expElogtheta[d, :]
#             expElogbetad = self._expElogbeta[:, ids]
#             # The optimal phi_{dwk} is proportional to 
#             # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
#             phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
#             # Iterate between gamma and phi until convergence
#             for it in range(0, 100):
#                 lastgamma = gammad
#                 # We represent phi implicitly to save memory and time.
#                 # Substituting the value of the optimal phi back into
#                 # the update for gamma gives this update. Cf. Lee&Seung 2001.
#                 gammad = self._alpha + expElogthetad * \
#                     n.dot(cts / phinorm, expElogbetad.T)
#                 Elogthetad = dirichlet_expectation(gammad)
#                 expElogthetad = n.exp(Elogthetad)
#                 phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
#                 # If gamma hasn't changed much, we're done.
#                 meanchange = n.mean(abs(gammad - lastgamma))
#                 if (meanchange < meanchangethresh):
#                     break
#             gamma[d, :] = gammad
#             # Contribution of document d to the expected sufficient
#             # statistics for the M step.
#             sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

#         # This step finishes computing the sufficient statistics for the
#         # M step, so that
#         # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
#         # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
#         sstats = sstats * self._expElogbeta

#         return((gamma, sstats))

    def update_lambda_docs(self, docs):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step_docs(docs)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound_docs(docs, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(docs))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        if self.save:
            lda_params = [self._rhot, self._lambda, self._Elogbeta, self._expElogbeta, self._updatect]
            file_name = self.file_name + '.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(lda_params, f)

        return(gamma, bound)

    def update_lambda(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(wordids, wordcts, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(wordids))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, wordids, wordcts, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             print oldphinorm
#             print n.log(phinorm)
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(wordids)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)

    def approx_bound_docs(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = parse_doc_list(docs, self._vocab)
        batchD = len(docs)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             print oldphinorm
#             print n.log(phinorm)
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)

        def set_file_name(self, file_name):
            self.file_name = file_name


def main():
    infile = sys.argv[1]
    K = int(sys.argv[2])
    alpha = float(sys.argv[3])
    eta = float(sys.argv[4])
    kappa = float(sys.argv[5])
    S = int(sys.argv[6])

    docs = corpus.corpus()
    docs.read_data(infile)

    vocab = open(sys.argv[7]).readlines()
    model = OnlineLDA(vocab, K, 100000,
                      0.1, 0.01, 1, 0.75)
    for i in range(1000):
        print i
        wordids = [d.words for d in docs.docs[(i*S):((i+1)*S)]]
        wordcts = [d.counts for d in docs.docs[(i*S):((i+1)*S)]]
        model.update_lambda(wordids, wordcts)
        n.savetxt('/tmp/lambda%d' % i, model._lambda.T)
    
#     infile = open(infile)
#     corpus.read_stream_data(infile, 100000)

#if __name__ == '__main__':
#    main()

"""
Corpus functions
for reading and organizing data
"""

class document:
    ''' the class for a single document '''
    def __init__(self):
        self.words = []
        self.counts = []
        self.length = 0
        self.total = 0

class corpus:
    ''' the class for the whole corpus'''
    def __init__(self):
        self.size_vocab = 0
        self.docs = []
        self.num_docs = 0

    def read_data(self, filename):
        if not os.path.exists(filename):
            print 'no data file, please check it'
            return
        print 'reading data from %s.' % filename

        for line in file(filename): 
            ss = line.strip().split()
            if len(ss) == 0: continue
            doc = document()
            doc.length = int(ss[0])

            doc.words = [0 for w in range(doc.length)]
            doc.counts = [0 for w in range(doc.length)]
            for w, pair in enumerate(re.finditer(r"(\d+):(\d+)", line)):
                doc.words[w] = int(pair.group(1))
                doc.counts[w] = int(pair.group(2))

            doc.total = sum(doc.counts) 
            self.docs.append(doc)

            if doc.length > 0:
                max_word = max(doc.words)
                if max_word >= self.size_vocab:
                    self.size_vocab = max_word + 1

            if (len(self.docs) >= 10000):
                break
        self.num_docs = len(self.docs)
        print "finished reading %d docs." % self.num_docs



def read_stream_data(f, num_docs):
  c = corpus()
  splitexp = re.compile(r'[ :]')
  for i in range(num_docs):
    line = f.readline()
    line = line.strip()
    if len(line) == 0:
      break
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    c.docs.append(d)

  c.num_docs = len(c.docs)
  return c

# This version is about 33% faster
def read_data(filename):
    c = corpus()
    splitexp = re.compile(r'[ :]')
    for line in open(filename):
        d = document()
        splitline = [int(i) for i in splitexp.split(line)]
        wordids = splitline[1::2]
        wordcts = splitline[2::2]
        d.words = wordids
        d.counts = wordcts
        d.total = sum(d.counts)
        d.length = len(d.words)
        c.docs.append(d)

        if d.length > 0:
            max_word = max(d.words)
            if max_word >= c.size_vocab:
                c.size_vocab = max_word + 1

    c.num_docs = len(c.docs)
    return c

def count_tokens(filename):
    num_tokens = 0
    splitexp = re.compile(r'[ :]')
    for line in open(filename):
        splitline = [int(i) for i in splitexp.split(line)]
        wordcts = splitline[2::2]
        num_tokens += sum(wordcts)

    return num_tokens

splitexp = re.compile(r'[ :]')
def parse_line(line):
    line = line.strip()
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    return d

