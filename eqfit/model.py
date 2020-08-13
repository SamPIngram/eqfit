from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
import numpy as np

class fitter():

    def __init__(self):
        self.X = None
        self.Y = None
        self.test_size = 0.2
        self.random_state = 0
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_type = 'PolyLasso'
        self.model = None

    def set_inputs(self, X):
        self.X = X

    def set_target(self, Y):
        self.Y = Y

    def train_test_split(self, test_size=0.2, random_state=0):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.test_size, random_state=self.random_state)

    def do_polyfit(self, degree_min=2, degree_max=4):
        # Alpha (regularization strength) of LASSO regression
        lasso_eps = 0.001
        lasso_nalpha=20
        lasso_iter=500000
        for degree in range(degree_min, degree_max+1):
            self.model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=10,n_jobs=-1))
            self.model.fit(self.X_train,self.y_train)
            test_pred = np.array(self.model.predict(self.X_test))
            RMSE=np.sqrt(np.sum(np.square(test_pred-self.y_test)))
            test_score = self.model.score(self.X_test,self.y_test)
            print(test_score)

    def display_equation(self, param_notation=None, coef_cutoff=0):
        coefs = self.model.named_steps['lassocv'].coef_
        eq_string = ''
        if param_notation is None:
            param_notation = [f'P{i}' for i in range(self.model.named_steps['polynomialfeatures'].n_input_features_)]
        assert len(param_notation) == len(self.model.named_steps['polynomialfeatures'].n_input_features_), 'Length of parameter notation does not match number of initial parameters.'
        for index, coeff in zip(((coefs > coef_cutoff)|(coefs < -coef_cutoff)).nonzero()[0], coefs[(coefs > coef_cutoff)|(coefs < -coef_cutoff)]):
            for n, (power, param) in enumerate(zip(self.model.named_steps['polynomialfeatures'].powers_[index], param_notation)):
                if n == 0:
                    if coeff > 0:
                        eq_string += f'+{round(coeff,4)}'
                    else:
                        eq_string += f'{round(coeff,4)}'
                if power != 0:
                    if power != 1:
                        eq_string += f'{param}^{power}'
                    else:
                        eq_string += f'{param}'
            eq_string += ' '
        print(eq_string)