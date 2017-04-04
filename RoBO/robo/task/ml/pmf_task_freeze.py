import numpy as np
import time
from abc import ABCMeta, abstractmethod
from robo.task.base_task import BaseTask
#from load_data import load_ml_1m, build_ml_1m, load_rating_matrix
#from matrix_factorization import MatrixFactorization
#from evaluation_metrics import RMSE
#from base import Base, DimensionError

"""
Adapted from https://github.com/chyikwei/recommend
"""


class NotImplementedError(Exception):

    def __init__(self):
        super(NotImplementedError, self).__init__()


class DimensionError(Exception):

    def __init__(self):
        super(DimensionError, self).__init__()


class Base(object):

    """base class"""

    __metaclass__ = ABCMeta

    def __init__(self):
        self.train_errors = []
        self.validation_erros = []

    @abstractmethod
    def estimate(self, iter=1000):
        """training models"""
        raise NotImplementedError

    @abstractmethod
    def suggestions(self, user_id, num=10):
        """suggest items for given user"""
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path):
        raise NotImplementedError

    @classmethod
    def load_model(cls, path):
        """load saved models"""
        raise NotImplementedError

    def __reper__(self):
        return self.__class__.__name__


class PMF(BaseTask):

    def __init__(self, train=None, train_targets=None,
                 valid=None, valid_targets=None,
                 test=None, test_targets=None,
                 n_classes=None, num_epochs=500,
                 save=False, file_name=None,
                 iteration_nr=10):
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
        self.iterations_nr = iteration_nr
        self.rec=None
        self.base_name = "pmf"

        # 1 Dim Learning Rate: 1e-4 to 1e-1
        # 2 Dim L2 regularization: 0 to 1
        X_lower = np.array([1e-4, 0.])
        X_upper = np.array([1e-1, 1.0])
        super(PMF, self).__init__(X_lower, X_upper)




    def set_weights(self, old_file_name):
        file_name = old_file_name# + '.pkl'
        self.rec.load_features(path=file_name)

    def set_epochs(self, n_epochs):
        self.num_epochs = n_epochs

    def set_iterations(self, iterations_nr):
        self.iterations_nr = iterations_nr

    def set_save_modus(self, is_old=True, file_old=None, file_new=None):
        self.save_old = is_old
        self.save = True
        if self.save_old:
            self.file_name = file_old
            if self.rec is not None:
                self.rec.set_save(True)
                self.rec.set_file_name(file_old)
        else:
            self.file_name = file_new
            if self.rec is not None:
                self.rec.set_save(True)
                self.rec.set_file_name(file_new)


    def objective_function(self, x):
        learning_rate = np.float32(np.exp(x[0, 0]))
        l2_reg = np.float32(x[0, 1])

        best_validation_loss = np.inf

        val_losses = []

        self.num_epochs=10

        ########################################################################################


        # load MovieLens data
        # num_user, num_item, ratings = load_ml_1m()
        num_user, num_item, ratings = build_ml_1m()
        np.random.shuffle(ratings)

        # set feature numbers
        num_feature = 10

        # set max_iterations
        max_iter = 20

        # split data to training & testing
        train_pct = 0.9
        #train_size = int(train_pct * len(self.ratings))
        train_size = int(train_pct * len(ratings))
        train = ratings[:train_size]
        validation = ratings[train_size:]

        # models
        self.rec = MatrixFactorization(num_user, num_item, num_feature, train, validation, max_rating=5, min_rating=1)
        self.rec.set_iterations(self.iterations_nr)
        if self.save:
            self.rec.set_save(True)
            self.rec.set_file_name(self.file_name)

        # fitting # here we should also receive the best_validation_loss and the val_losses
        best_validation_loss, val_losses = self.rec.estimate(max_iter)

        return best_validation_loss, val_losses
        
    def objective_function_test(self, x):        
        self.objective_function(x)
        return self.test_error



class MatrixFactorization(Base):

    def __init__(self, num_user, num_item, num_feature, train, validation, **params):
        super(MatrixFactorization, self).__init__()
        self._num_user = num_user
        self._num_item = num_item
        self._num_feature = num_feature
        self.train = train
        self.validation = validation

        # batch size
        self.batch_size = int(params.get('batch_size', 100000))

        # learning rate
        self.epsilon = float(params.get('epsilon', 100.0))
        # regularization parameter (lambda)
        self.lam = float(params.get('lam', 0.001))

        self.max_rating = params.get('max_rating')
        self.min_rating = params.get('min_rating')
        if self.max_rating:
            self.max_rating = float(self.max_rating)
        if self.min_rating:
            self.min_rating = float(self.min_rating)

        # mean rating
        self._mean_rating = np.mean(self.train[:, 2])

        # latent variables
        self._user_features = 0.3 * np.random.rand(num_user, num_feature)
        self._item_features = 0.3 * np.random.rand(num_item, num_feature)

    @property
    def user(self):
        return self._num_user

    @property
    def items(self):
        return self._num_item

    @property
    def user_features(self):
        return self._user_features

    def set_iterations(self, iterations_nr):
        self.iterations_nr = iterations_nr

    def set_save(self, save):
        self.save = save

    def set_file_name(self, file_name):
        self.file_name=file_name

    @user_features.setter
    def user_features(self, val):
        old_shape = self._user_features.shape
        new_shape = val.shape
        if old_shape == new_shape:
            self._user_features = val
        else:
            raise DimensionError()

    @property
    def item_features(self):
        return self._item_features

    @item_features.setter
    def item_features(self, val):
        old_shape = self._item_features.shape
        new_shape = val.shape
        if old_shape == new_shape:
            self._item_features = val
        else:
            raise DimensionError()

    def estimate(self, iterations=50, converge=1e-4):
        last_rmse = None
        batch_num = int(np.ceil(float(len(self.train) / self.batch_size)))
        best_validation_loss = np.inf
        print "batch_num =", batch_num + 1
        runs = self.iterations_nr

        for iteration in xrange(runs):
            # np.random.shuffle(self.train)

            for batch in xrange(batch_num):
                data = self.train[
                    batch * self.batch_size: (batch + 1) * self.batch_size]
                # print "data", data.shape

                # compute gradient
                u_features = self._user_features[data[:, 0], :]
                i_features = self._item_features[data[:, 1], :]
                # print "u_feature", u_features.shape
                # print "i_feature", i_features.shape
                ratings = data[:, 2] - self._mean_rating
                preds = np.sum(u_features * i_features, 1)
                errs = preds - ratings
                err_mat = np.tile(errs, (self._num_feature, 1)).T
                # print "err_mat", err_mat.shape

                u_grads = u_features * err_mat + self.lam * i_features
                i_grads = i_features * err_mat + self.lam * u_features

                u_feature_grads = np.zeros((self._num_user, self._num_feature))
                i_feature_grads = np.zeros((self._num_item, self._num_feature))

                for i in xrange(data.shape[0]):
                    user = data[i, 0]
                    item = data[i, 1]
                    u_feature_grads[user, :] += u_grads[i,:]
                    i_feature_grads[item, :] += i_grads[i,:]

                # update latent variables
                self._user_features = self._user_features - \
                    (self.epsilon / self.batch_size) * u_feature_grads
                self._item_features = self._item_features - \
                    (self.epsilon / self.batch_size) * i_feature_grads

            # compute RMSE
            # train errors

            train_preds = self.predict(self.train)
            train_rmse = RMSE(train_preds, np.float16(self.train[:, 2]))

            # validation errors
            validation_preds = self.predict(self.validation)
            validation_rmse = RMSE(
                validation_preds, np.float16(self.validation[:, 2]))
            if validation_rmse < best_validation_loss:
                best_validation_loss = validation_rmse

            self.train_errors.append(train_rmse)
            self.validation_erros.append(validation_rmse)
            print "iterations: %3d, train RMSE: %.6f, validation RMSE: %.6f " % \
                (iteration + 1, train_rmse, validation_rmse)

            if self.save:
                self.save_features(path=self.file_name)

            return np.array([[best_validation_loss]]), self.validation_erros

    def predict(self, data):
        u_features = self._user_features[data[:, 0], :]
        i_features = self._item_features[data[:, 1], :]
        preds = np.sum(u_features * i_features, 1) + self._mean_rating

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def suggestions(self, user_id, num=10):
        # TODO
        pass

    def save_model(self):
        # TODO
        pass

    def load_model(self):
        # TODO
        pass

    def load_features(self, path):
        import cPickle
        import gzip
        with gzip.open(path, 'rb') as f:
            self._user_features = cPickle.load(f)
            self._item_features = cPickle.load(f)
            num_user, num_feature_u = self._user_features.shape
            num_item, num_feature_i = self._item_features.shape

            if num_feature_i != num_feature_u:
                raise DimensionError()
            self._num_feature = num_feature_i

        return self

    def save_features(self, path):
        import cPickle
        import gzip
        with gzip.open(path, 'wb') as f:
            cPickle.dump(
                self._user_features, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(
                self._item_features, f, protocol=cPickle.HIGHEST_PROTOCOL)





def RMSE(estimation, truth):
    """Root Mean Square Error"""

    num_sample = len(estimation)

    # sum square error 
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1.0))

"""
load data set

"""


def build_ml_1m():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """
    num_user = 6040
    num_item = 3952
    print("\nloadind movie lens 1M data")
    with open("data/ratings.dat", "rb") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line.split('::')[:3]
            line = [int(l) for l in line]
            ratings.append(line)

            if line_num % 100000 == 0:
                print line_num

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)] - 1
    print "max user id", max(ratings[:, 0])
    print "max item id", max(ratings[:, 1])
    return num_user, num_item, ratings


def load_ml_1m():
    """load Movie Lens 1M ratings from saved gzip file"""
    import gzip
    import cPickle

    file_path = 'data/ratings.gz'
    with gzip.open(file_path, 'rb') as f:
        print "load ratings from: %s" % file_path
        num_user = cPickle.load(f)
        num_item = cPickle.load(f)
        ratings = cPickle.load(f)

        return num_user, num_item, ratings


def build_rating_matrix(num_user, num_item, ratings):
    """
    build dense ratings matrix from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print '\nbuild matrix'
    # sparse matrix
    #matrix = sparse.lil_matrix((num_user, num_item))
    # dense matrix
    matrix = np.zeros((num_user, num_item), dtype='int8')
    for item_id in xrange(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            matrix[data[:, 0], item_id] = data[:, 2]

        if item_id % 1000 == 0:
            print item_id

    return matrix


def load_rating_matrix():
    """
    load Movie Lens 1M ratings from saved gzip file
    Format is numpy dense matrix
    """
    import gzip
    import cPickle

    file_path = 'data/rating_matrix.gz'
    with gzip.open(file_path, 'rb') as f:
        print "load ratings matrix from: %s" % file_path
        return cPickle.load(f)


def build_sparse_matrix(num_user, num_item, ratings):
    # TODO: have not tested it yet. will test after algorithm support sparse
    # matrix
    print '\nbuild sparse matrix'
    # sparse matrix
    matrix = sparse.lil_matrix((num_user, num_item))
    for item_id in xrange(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            # for sparse matrix
            matrix[data[:, 0], item_id] = np.array([data[:, 2]]).T

        if item_id % 1000 == 0:
            print item_id

    return matrix