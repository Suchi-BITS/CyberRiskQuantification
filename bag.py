import networkx as nx
#from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
#from pgmpy.factors.discrete import TabularCPD

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def build_attack_graph():
    # Define the structure of the Bayesian Network
    model = DiscreteBayesianNetwork([
        ('VulnA', 'Compromise1'),
        ('VulnB', 'Compromise1'),
        ('Compromise1', 'Compromise2')
    ])

    # Define the CPDs (Conditional Probability Distributions)
    cpd_vulnA = TabularCPD(variable='VulnA', variable_card=2, values=[[0.7], [0.3]])
    cpd_vulnB = TabularCPD(variable='VulnB', variable_card=2, values=[[0.6], [0.4]])
    cpd_compromise1 = TabularCPD(variable='Compromise1', variable_card=2,
                                 values=[[0.9, 0.5, 0.8, 0.3],
                                         [0.1, 0.5, 0.2, 0.7]],
                                 evidence=['VulnA', 'VulnB'],
                                 evidence_card=[2, 2])
    cpd_compromise2 = TabularCPD(variable='Compromise2', variable_card=2,
                                 values=[[0.95, 0.2],
                                         [0.05, 0.8]],
                                 evidence=['Compromise1'],
                                 evidence_card=[2])

    # Add CPDs to the model
    model.add_cpds(cpd_vulnA, cpd_vulnB, cpd_compromise1, cpd_compromise2)

    # Check if the model is valid
    assert model.check_model()

    return model



def infer_attack_probabilities(model):
    infer = VariableElimination(model)
    query_result = infer.query(variables=['Compromise2'], evidence={'VulnA': 1})
    print("Probability of Compromise2 given VulnA=1:")
    print(query_result)


if __name__ == "__main__":
    graph_model = build_attack_graph()
    infer_attack_probabilities(graph_model)
