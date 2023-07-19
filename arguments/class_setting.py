


class ClassSetting(object):
    """ Class to hold parameters of classification problem

    Attributes
    ----------
    confidence_scores_class : dict {string : float}
        Dictionary holding per methylation class a minimum confidence score will be used if filter_methylation_confidence is True
    confidence_scores_subclass : dict {string : float}
        Dictionary holding per methylation subclass a minimum confidence score will be used if filter_methylation_confidence is True
    class_attribute : string
        Name of column in .csv to use for classification problem
    class_labels : dict {string : int}
        Class label per used value of classification attribute - must hold all values starting from 0
    n_classes : int
        Number of classes in classification problem
    filter_inconsistent_methylation : bool
        True if want to filter patients with inconsistent methylation class/subclass combination as defined in possible_methylation_classes
    filter_methylation_confidence : bool
        True if want to filter patients with too low confidence score in methylation class as defined in confidence_scores_class and confidence_scores_subclass
    possible_methylation_classes : dict {string : [string]}
        Possible combinations of methylation class and subclass

    Methods
    -------
    get_confidence_scores_class()
        Return confidence_scores_class
    get_confidence_scores_subclass()
        Return confidence_scores_subclass
    get_class_attribute()
        Return class_attribute
    get_class_labels()
        Return class_labels
    get_n_classes()
        Return n_classes
    get_filter_inconsistent_methylation()
        Return filter_inconsistent_methylation
    get_filter_methylation_confidence()
        Return filter_methylation_confidence
    get_possible_methylation_classes()
        Return possible_methylation_classes
    """
    def __init__(self):
        # Confidence scores for filtering per class        
        self.confidence_scores_class = {'Meningioma benign':0.5, 
        'Meningioma intermediate':0.5,
        'Meningeoma Malignant':0.5}
        # Confidence scores for filtering per subclass
        self.confidence_scores_subclass = {'Meningioma benign-1':0.5,
        'Meningioma benign-2':0.5,
        'Meningioma benign-3':0.5,
        'Meningioma intermediate-A':0.5,
        'Meningioma intermediate-B':0.5,
        'Meningeoma Malignant':0.5}
        # Class attribute for classification
        self.class_attribute = "Methylation class family member (EPIC) Meningioma-Classifier"
        # Class labels according to class attribute - determines valid samples
        self.class_labels = {
            "Meningioma benign-1":0,
            "Meningioma benign-2":1
        }
        # Number of classes is highest label + 1
        self.n_classes = max(self.class_labels.values()) + 1

        # Set True if want to filter patients with inconsistent class/subclass
        self.filter_inconsistent_methylation = True
        # Set True if want to filter patients with too low confidence scores in EPIC classification
        self.filter_methylation_confidence = True

        # possible class/subclass combinations
        self.possible_methylation_classes = {
        "Meningioma":["Meningioma"],
        "Meningioma benign":["Meningioma benign-1", "Meningioma benign-2", "Meningioma benign-3"],
        "Meningioma intermediate":["Meningioma intermediate-A", "Meningioma intermediate-B"],
        "Meningeoma Malignant":["Meningeoma Malignant"],
        "?":"?"}


    def get_confidence_scores_class(self):
        """ Return confidence_scores_class
        Returns
        -------
        dict {string : float}
            confidence_scores_class
        """
        return self.confidence_scores_class

    def get_confidence_scores_subclass(self):
        """ Return confidence_scores_subclass
        Returns
        -------
        dict {string : float}
            confidence_scores_subclass
        """
        return self.confidence_scores_subclass

    def get_class_attribute(self):
        """ Return class_attribute
        Returns
        -------
        string
            class_attribute
        """
        return self.class_attribute

    def get_class_labels(self):
        """ Return class_labels
        Returns
        -------
        dict {string : int}
            class_labels
        """
        return self.class_labels

    def get_n_classes(self):
        """ Return n_classes
        Returns
        -------
        int
            n_classes
        """
        return self.n_classes

    def get_filter_inconsistent_methylation(self):
        """ Return filter_inconsistent_methylation
        Returns
        -------
        bool
            filter_inconsistent_methylation
        """
        return self.filter_inconsistent_methylation

    def get_filter_methylation_confidence(self):
        """ Return filter_methylation_confidence
        Returns
        -------
        bool
            filter_methylation_confidence
        """
        return self.filter_methylation_confidence

    def get_possible_methylation_classes(self):
        """ Return possible_methylation_classes
        Returns
        -------
        dict {string : [string]}
            possible_methylation_classes        
        """
        return self.possible_methylation_classes