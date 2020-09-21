import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


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
        self.polymodel = None
        self.lassomodel = None
        self.model = None
        self.param_notation = None
        self.equation = None

    def set_inputs(self, X):
        """Input parameters for the equation you want to make. Parameters with little influence can be removed in the make_equation method.

        Args:
            X (pandas.Dataframe): Dataframe with the target prediction column removed. Can use any number of input parameters.
        """
        self.X = X

    def set_target(self, Y):
        """Input target (prediction) for the equation you want to make.

        Args:
            Y (pandas.Dataframe): Dataframe of only the target prediction column. Should be a single column.
        """
        self.Y = Y

    def train_test_split(self, test_size=0.2, random_state=0):
        """Splits the data for validation of polyfit equations. This data is used automatically during the do_polyfit method.

        Args:
            test_size (float, optional): Sets the fraction of data which will be put aside to perform validation on and get the score during the do_polyfit method. Defaults to 0.2.
            random_state (int, optional): Sets a random seed for reproducible splitting of the data. Defaults to 0.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.test_size, random_state=self.random_state)

    def do_polyfit(self, degree_min=2, degree_max=4, lasso_eps=0.001, lasso_nalpha=20, lasso_iter=500000, verbose=False):
        """Carries out testing how well input parameters can fit the target parameters using a LASSO regression. This will iteratively test from degree_min to degree_max the score (adjusted R^2) that can be achieved.
        Only the best performing poolynomial fit will be saved.

        Args:
            degree_min (int, optional): Minium polynomail degree fit tested. Defaults to 2.
            degree_max (int, optional): Maxium polynomail degree fit tested. Defaults to 4.
            lasso_eps (float, optional): Lasso arg - Length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3. Defaults to 0.001.
            lasso_nalpha (int, optional): Lasso arg - Number of alphas along the regularization path. Defaults to 20.
            lasso_iter (int, optional): Lasso arg - The maximum number of iterations. Defaults to 500000.
            verbose (bool, optional): Prints out scores for every degree tested. Defaults to False.
        """
        # Alpha (regularization strength) of LASSO regression
        lasso_eps = lasso_eps
        lasso_nalpha = lasso_nalpha
        lasso_iter = lasso_iter
        top_score = 0
        for degree in range(degree_min, degree_max+1):
            model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=10,n_jobs=-1))
            model.fit(self.X_train,self.y_train)
            test_pred = np.array(model.predict(self.X_test))
            RMSE=np.sqrt(np.sum(np.square(test_pred-self.y_test)))
            r_squared = model.score(self.X_test,self.y_test)
            adjusted_r_squared = 1 - (1-r_squared)*(len(self.y_test)-1)/(len(self.y_test)-self.X_test.shape[1]-1)
            if verbose:
                print(f'Degree: {degree}, RMSE: {round(RMSE,5)}, R^2(adj): {round(adjusted_r_squared,5)}, Features: {model.named_steps["polynomialfeatures"].n_output_features_}')
            if adjusted_r_squared > top_score:
                top_score = adjusted_r_squared
                self.polymodel = model.named_steps['polynomialfeatures']
                self.lassomodel = model.named_steps['lassocv']
                self.degree = degree
                self.model = model

    def make_equation(self, param_notation=None, coef_cutoff=0, use_effective_coef=True):
        """Examines the coefficients for every feature in the polynomial fit and produces an equation. 

        Args:
            param_notation (list, optional): Give the desired notation for the resultant equation, if None this will simply use P1, P2... Pn. Defaults to None.
            coef_cutoff (float, optional): Chose a coefficient threshold to ignore terms which contribute little to the overall result. Defaults to 0.
            use_effective_coef (bool, optional): Accounts for coefficient on high polynomial terms being small. This will balance coefficients between different degree multiplication. Defaults to True.
        """
        coefs = self.model.named_steps['lassocv'].coef_
        param_norm = [self.X[col].max() for col in self.X.columns]
        eq_string = ''
        if param_notation is None:
            param_notation = [f'P{i}' for i in range(self.model.named_steps['polynomialfeatures'].n_input_features_)]
        assert len(param_notation) == self.model.named_steps['polynomialfeatures'].n_input_features_, 'Length of parameter notation does not match number of initial parameters.'
        for index, coeff in enumerate(coefs):
            if use_effective_coef is False and coeff < coef_cutoff and coeff > -coef_cutoff:
                continue
            else:
                powers = self.model.named_steps['polynomialfeatures'].powers_[index]
                effective_coeff = coeff
                for power, param_max in zip(powers, param_norm):
                    effective_coeff *= param_max**power
                if effective_coeff < coef_cutoff and effective_coeff > -coef_cutoff:
                    continue
            for n, (power, param) in enumerate(zip(self.model.named_steps['polynomialfeatures'].powers_[index], param_notation)):
                if n == 0:
                    eq_string += f'+ ({format(coeff, "10.2E")}*'
                if power != 0:
                    if power != 1:
                        eq_string += f'({param}**{power})*'
                    else:
                        eq_string += f'({param})*'
            eq_string = eq_string[:-1]  # Deletes additional * from last term per coefficient entry
            eq_string += ') ' # Adds brackets for easier reading
        self.equation = eq_string
        self.param_notation = param_notation
        print(eq_string)

    def display_equation(self):
        """Prints the equation from the make_equation method to screen.
        """
        print(self.equation)


    def evaluate_equation(self, plot=True):
        """Evaluate how the made equation from the make_equation method performs compared to actual given target prediction values.

        Args:
            plot (bool, optional): Plot a graph of equation predictions vs actual values set in the set_target method. Defaults to True.
        """
        print(f'Evaluating: {self.equation}')
        predictions = []
        for i, row in self.X.iterrows():
            for param, col in zip(self.param_notation, self.X.columns):
                globals()[param] = row[col]
            predictions.append(eval(self.equation))
        RMSE=np.sqrt(np.sum(np.square(predictions-self.Y)))
        print(f'RMSE: {round(RMSE,5)}')
        print(f'RMSE/datapoint: {round(RMSE,5)/len(predictions)}')
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(x=self.Y, y=predictions)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            # now plot both limits against eachother
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_ylabel('Predicted')
            ax.set_xlabel('Actual')
            plt.tight_layout()
            plt.show()
