###################################################
##             ASU CSE 571 ONLINE                ##
##        Unit 3 Reasoning under Uncertainty     ##
##             Project Submission File           ##
##                    agent.py                   ##
###################################################

###################################################
##                !!!IMPORTANT!!!                ##
##        This file will be auto-graded          ##
##    Do NOT change this file other than at the  ##
##       Designated places with your code        ##
##                                               ##
##  READ the instructions provided in the code   ##
###################################################

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

def buildDBN():

    # Construct a DBN object
    dbn = DBN()

    dbn.add_edges_from([(('L', 0), ('O', 0)), (('L', 1), ('O', 1)), (('L', 0), ('L', 1))])
    l_cpd = TabularCPD(('L', 0), 4, [[0], [0], [1],[0]])
    o_cpd = TabularCPD(('O', 0), 4, [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]], evidence=[('L', 0)], evidence_card=[4])

    l_i_cpd = TabularCPD(('L', 1), 4, [[0.5, 0, 0.5, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0,0.5, 0, 0.5]], evidence=[('L', 0)], evidence_card=[4])
    # Do NOT forget to initialize before doing any inference! Otherwise, errors will be introduced.

    dbn.add_cpds(l_cpd, o_cpd, l_i_cpd)


    # Do NOT forget to initialize before doing any inference! Otherwise, errors will be introduced.
    dbn.initialize_initial_state()

    # Create an inference object
    dbn_inf = DBNInference(dbn)
    #print(dbn_inf.query(variables=[('L',1)], evidence={ ('O',1): 2})[('L',1)])
    return dbn_inf
