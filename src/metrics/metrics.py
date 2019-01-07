from sklearn.metrics import *

class unweighted_recall(recall_score):

     def __init__(self, dtype='float', **kwargs):
        super().__init__(**kwargs)

     
