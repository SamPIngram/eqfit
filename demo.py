import eqfit
import pandas as pd

data = pd.read_csv('example/exampledata.csv')

eq = eqfit.fitter()
eq.set_inputs(data.drop(columns=['Y']))
eq.set_target(data['Y'])
eq.train_test_split()
eq.do_polyfit(verbose=True)
eq.make_equation(param_notation=['A', 'B', 'C', 'D', 'E'], coef_cutoff=100)
eq.evaluate_equation()