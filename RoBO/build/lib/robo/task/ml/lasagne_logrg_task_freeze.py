import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params, l2, l1

from robo.task.base_task import BaseTask


class LogisticRegression(BaseTask):

    def __init__(self, train, train_targets,
                 valid, valid_targets,
                 test, test_targets,
                 n_classes, num_epochs=500,
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
        self.base_name = "logistic_regression"
        # 1 Dim Learning Rate:
        # 2 Dim L2 regularization: 0 to 1
        # 3 Dim Batch size: 20 to 2000
        # 4 Dim Dropout rate: 0 to 0.75
        # 5 Dim L1 regularization: 0.1 to 20
        # 6 Dim Epochs Number: 1 to 100
        X_lower = np.array([np.log(1e-6), 0.0, 20, 0, 0.1, 1])
        #X_lower = np.array([np.log(1e-6), 0.0, 1000, 0, 0.1, 1])
        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 100])
        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 10])
        X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 7])
        #X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 3])
        super(LogisticRegression, self).__init__(X_lower, X_upper)


    def build_mlp(self, input_var=None, dropout_rate=0.5, l2_reg=0., l1_reg=0.):

        l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                         input_var=input_var)

        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=dropout_rate)

       

        self.l2_penalty = regularize_layer_params(l_in_drop, l2)
        self.l1_penalty = regularize_layer_params(l_in_drop, l1)

        


        l_out = lasagne.layers.DenseLayer(
                l_in_drop, num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)


        return l_out

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

        #print 'batchsize: ', batchsize
        #print 'len(inputs): ', len(inputs)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def set_weights(self, old_file_name):
        file_name = old_file_name + '.npz'
        with np.load(file_name) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)

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
        print 'in objective_function x: ', x
        learning_rate = np.float32(np.exp(x[0, 0]))
        l2_reg = np.float32(x[0, 1])
        batch_size = np.int32(x[0, 2])
        print 'in objective_function batch_size: ', batch_size
        dropout_rate = np.int32(x[0, 3])
        l1_reg = np.int32(x[0, 4])
        epochs_number = np.int32(x[0, 5])
        best_validation_loss = np.inf
        self.num_epochs = epochs_number

        val_losses = []

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        #input_var = T.matrix('inputs')
        target_var = T.ivector('targets')

        # Create neural network model (depending on first command line parameter)
        print("Building model and compiling functions...")
        self.network = self.build_mlp(input_var, dropout_rate, l2_reg, l1_reg)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()


        #l2_penalty = regularize_layer_params_weighted(layers, l2)
        #l1_penalty = regularize_layer_params(layer2, l1) * 1e-4
        loss = loss + l2_reg*self.l2_penalty + l1_reg*self.l1_penalty
        # We could add some weight decay as well here, see lasagne.regularization.

        # Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers more.
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=learning_rate, momentum=0.9)

        # Create a loss expression for validation/testing. The difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(self.num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(self.X_train, self.y_train, batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(self.X_val, self.y_val, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            val_loss = 1. - val_acc / float(val_batches)
            val_losses.append(val_loss)
            
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            print("  best_validation_loss:\t\t{:.6f}".format(best_validation_loss))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.iterate_minibatches(self.X_test, self.y_test, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1


        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))


        if self.save:
            file_name = self.file_name + '.npz'
            np.savez(file_name, *lasagne.layers.get_all_param_values(self.network))

        return np.array([[best_validation_loss]]), val_losses
        
    def objective_function_test(self, x):        
        self.objective_function(x)
        return self.test_error