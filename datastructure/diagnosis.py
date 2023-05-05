import numpy as np

class Diagnosis(object):
    """
    A class to represent the diagnosis given for one patient

    Attributes
    ----------
    class_attribute : string
        Value in classification attribute
    methylation_class : string
        Methylation class as obtained by EPIC
    methylation_class_score : float
        Methylation class probability score as obtained by EPIC
    methylation_subclass : string
        Methylation subclass as obtained by EPIC
    methylation_subclass_score : float
        Methylation subclass probability score as obtained by EPIC
    consistent : bool
        True if methylation class and subclass form a valid tuple
    label : int
        Label according to class attribute and data setting. None if not set.
    who_grade : string
        Histological diagnosis according to WHO criteria
    stamp_blocks : list
        List of blocks where DNA was extracted from 
    stamp_subblocks : list
        List of subblocks where DNA was extracted from - meant to be combined with stamp_blocks
    predicted_scores : list
        List of predicted scores for correct class for patient

    Methods
    -------
    check_consistency(setting)
        Checks methylation class / subclass validity
    get_class_attribute()
        Getter classification attribute value
    get_methylation_class()
        Getter methylation class
    get_methylation_class_score()
        Getter methylation class probability score
    get_methylation_subclass()
        Getter methylation subclass
    get_methylation_subclass_score()
        Getter methylation subclass probability score
    is_consistent()
        Getter of consistent attribute
    set_label(label)
        Set classification label for diagnosis
    get_label()
        Getter classification label
    get_who_grade()
        Getter who grade
    get_stamp_blocks()
        Getter stamp blocks
    get_stamp_subblocks()
        Getter stamp subblocks
    add_predicted(score)
        Append a prediction score to list of predicted scores
    save_predicted_scores(identifier, folder)
        Save predicted scores as .npy
    """
    def __init__(self, setting, class_attribute, methylation_class, methylation_class_score, methylation_subclass, methylation_subclass_score, who_grade, stamp_blocks, stamp_subblocks):
        """
        Parameters
        ----------
        setting : Setting  
            A Setting object as specified by the class
        class_attribute : string
            The value for the classification attribute
        methylation_class : string
            The methylation class of the patient
        methylation_class_score : float
            The probability score of the methylation class
        methylation_subclass : string
            The methylation subclass of the patient
        methylation_subclass_score : float
            The probability score of the methylation subclass
        who_grade : string
            The histological diagnosis of the patient according to WHO criteria
        stamp_blocks : list
            A list with the identifiers of the blocks where DNA was extracted from
        stamp_subblocks : list
            A list with the identifiers of the subblocks where DNA was extracted from - is thought to be combined with stamp_blocks
        """
        # Classification attribute
        self.class_attribute = class_attribute

        # patient tumor class and confidence e.g. 'Meningioma benign'
        self.methylation_class = methylation_class
        self.methylation_class_score = methylation_class_score
        
        # patient tumor subclass and confidence e.g. 'Meningioma bengin-1'
        self.methylation_subclass = methylation_subclass
        self.methylation_subclass_score = methylation_subclass_score

        # check if diagnosis is consistent
        self.consistent = self.check_consistency(setting)

        # label for diagnosis / class refering to diagnosis
        self.label = None
        
        # WHO Grade according to cns
        self.who_grade = who_grade
        
        # List of Blocks stamp taken from
        self.stamp_blocks = stamp_blocks
        # Subblock for stamp same order as block
        self.stamp_subblocks = stamp_subblocks


        # Predicted scores for this diagnosis
        self.predicted_scores = []

    def check_consistency(self, setting):
        """ Checks wheter methylation class and subclass are a valid combination as defined in class setting

        Parameters
        ----------
        setting : Setting
            A Setting object as specified by the class
        
        Returns
        -------
        bool
            True if valid combination False otherwise
        """
        # check if subclass is part of class
        return self.methylation_subclass in setting.get_class_setting().get_possible_methylation_classes()[self.methylation_class]
    
    def get_class_attribute(self):
        """ Getter of value for classification attribute

        Returns
        -------
        string
            Value for class attribute
        """
        return self.class_attribute

    def get_methylation_class(self):
        """ Getter for methylation class

        Returns
        -------
        string
            Methylation class
        """
        return self.methylation_class

    def get_methylation_class_score(self):
        """ Getter for methylation class probability score

        Returns
        -------
        float
            probability score for methylation class
        """
        return self.methylation_class_score

    def get_methylation_subclass(self):
        """ Getter for methylation subclass

        Returns
        -------
        string
            Methylation subclass
        """
        return self.methylation_subclass

    def get_methylation_subclass_score(self):
        """ Getter for methylation subclass probability score

        Returns
        -------
        float
            probability score for methylation subclass
        """
        return self.methylation_subclass_score

    def is_consistent(self):
        """ Getter for consitency of methylation class and subclass

        Returns
        -------
        bool 
            Consitentce of class and subclass
        """
        return self.consistent

    def set_label(self, label):
        """ Set label for classification problem

        Parameters
        ----------
        label : int
            Label for class attribute in current classification problem
        """
        self.label = label

    def get_label(self):
        """ Getter for label

        Returns
        -------
        int
            Label for class attribute
        """
        return self.label
    
    def get_who_grade(self):
        """ Getter for WHO grade

        Returns
        -------
        string
            Classification according to WHO
        """
        return self.who_grade

    def get_stamp_blocks(self):
        """ Getter for stamp blocks
        
        Returns
        -------
        list
            List of block identifiers
        """
        return self.stamp_blocks
    
    def get_stamp_subblocks(self):
        """ Getter for stamp subblocks

        Returns
        -------
        list 
            List of subblock identifiers
        """
        return self.stamp_subblocks

    def add_predicted_score(self, predicted_score):
        """ Appends a predicted score for this diagnosis

        Parameters
        ----------
        predicted_score : float
            Predicted score for this diagnosis
        """
        self.predicted_scores.append((predicted_score))

    def save_predicted_scores(self, identifier, folder):
        """ Save the predicted scores of this diagnosis to a .npy file with the patient identifier as file name

        Parameters
        ----------
        identifier : string
            The patients identifier
        folder : string
            The path where the scores should be saved to
        """
        #Added folder layer MD
        fn = folder +'/'+ identifier + '.npy'
        np.save(fn, np.array(self.predicted_scores))