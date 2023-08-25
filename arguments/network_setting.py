from utils import helper
class NetworkSetting(object):
    """ Class to store parameters used for model/training

    Attributes
    ----------
    batch_size : int
        Batch size to use in attention model training
    train_model : bool
        True if want to train the model
    model_folder : string
        Folder to store model parameter files to
    dropout : float
        Dropout rate in attention model 0 if no dropout should be used
    F : int
        Feature dimension
    L : int
        Next encoding stage size
    D : int
        Second next encoding stage size
    early_stopping : bool
        True if want to use early stopping
    patience : int
        Number of epochs without performance up to wait before EarlyStopping
    stop_epoch : int
        Minimum number of epochs to train model before EarlyStopping might be applied
    verbose : bool
        True if want to print console information 
    epochs : int 
        Number of epochs to train
    runs : int 
        Number of Monte Carlo runs

    Methods
    -------
    get_batch_size()
        Return batch_size
    get_train_model()
        Return train_model
    get_model_folder()
        Return model_folder
    get_dropout()
        Return dropout
    get_F()
        Return F
    get_L()
        Return L
    get_D()
        Return D
    get_early_stopping()
        Return early_stopping
    get_patience()
        Return patience
    get_stop_epoch()
        Return stop_epoch
    get_verbose()
        Return verbose
    get_epochs()
        Returns epochs
    get_runs()
        Return runs
    """
    def __init__(self, working_directory, json_file=None):
        """
        Parameters
        ----------
        working_directory : string
            Working directory for data
        json_file : dict or None
            contains a setup dict used for evaluation purposes
        """
        self.batch_size = 1

        self.train_model = True

        self.model_folder = working_directory+'Models/'

        #MD
        if json_file is not None:
            self.model_folder = json_file["test_score_folder"]+'/Models/'
        helper.create_folder(self.model_folder)

        self.dropout = 0.25
        

        self.F = 1024
        self.L = 512
        self.D = 384

        self.early_stopping = True
        self.patience = 25
        self.stop_epoch = 100

        self.verbose = True

        self.epochs = 3

        self.runs = 1

        if json_file is not None:
            self.epochs = json_file['epochs']
            self.runs = json_file['runs']

    #MD
    def reset_folder_path(self, settings):
        self.model_folder = settings.json_file["test_score_folder"]+'/Models/'
        helper.create_folder(self.model_folder)

    def get_batch_size(self):
        """ Return batch_size attribute
        Returns
        -------
        int 
            batch_size
        """
        return self.batch_size

    def get_train_model(self):
        """ Return train_model attribute
        Returns
        -------
        bool
            train_model
        """
        return self.train_model

    def get_model_folder(self):
        """ Return model_folder attribute and create folder if not existing
        Returns
        -------
        string
            model_folder
        """
        helper.create_folder(self.model_folder)
        return self.model_folder

    def get_dropout(self):
        """ Return dropout attribute
        Returns
        -------
        float
            dropout
        """
        return self.dropout
    
    def get_F(self):
        """ Return F attribute
        Returns
        -------
        int 
            F
        """
        return self.F

    def get_L(self):
        """ Returns L attribute
        Returns
        -------
        int
            L
        """
        return self.L
    
    def get_D(self):
        """ Returns D attribute
        Returns
        -------
        int
            D
        """
        return self.D

    def get_early_stopping(self):
        """ Returns early_stopping attribute
        Returns
        -------
        bool
            early_stopping
        """
        return self.early_stopping

    def get_patience(self):
        """ Return patience attribute
        Returns
        -------
        int
            patience
        """
        return self.patience

    def get_stop_epoch(self):
        """ Return stop_epoch attribute
        Returns
        -------
        int 
            stop_epoch
        """
        return self.stop_epoch

    def get_verbose(self):
        """ Return verbose attribute
        Returns
        -------
        bool
            verbose
        """
        return self.verbose
    
    def get_epochs(self):
        """ Return epochs attribute
        Returns
        -------
        int
            epochs
        """
        return self.epochs

    def get_runs(self):
        """ Return runs attribute
        Returns
        -------
        int
            runs
        """
        return self.runs