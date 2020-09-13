# eqfit - Equation fitting automation made simple with python.

Note: this is an early evaluation of what such a package might look like. If you are interested in contributing and/or guiding the direction of this project please feel free to reach out.

## Installation

eqfit can be installed using pip:
```
pip install eqfit
```

To get started quickly from source follow these steps:

1. Clone or download this repository and launch python within the folder.

2. Make a python environment to run in. Always recommend the use of virtual environments for safety. Must be Python3 and currently only tested with Python 3.8

3. Install requirement.txt file into your new python environment
```
pip install -r requirements.txt
```

4. Test the module is working on the example dataset using the demo.py script in the example directory.
```
python demo.py
``` 
If everything worked you should get the following figure output:

![demofig](https://github.com/SamPIngram/eqfit/blob/master/example/demofig.png?raw=true)

## Example

This example will cover the API to call eqfit on your data. 
It will utilise the [example dataset](https://github.com/SamPIngram/eqfit/blob/master/example/exampledata.csv) provided.
In this example, there are five input parameters (A-E) and one test parameter (Y). 
We want to figure out a suitable polynomial equation for predicting Y from the parameters A-E.
The below is a heavily commented version of the [demo python script](https://github.com/SamPIngram/eqfit/blob/master/demo.py) to explain how we do this.
```python
import eqfit # import eqfit
import pandas as pd # import pandas to load data
# example data loaded you can replace this with the data you wish to fit to.
# this does not have to be a .csv, but you will need to get it in a pandas dataframe.
data = pd.read_csv('example/exampledata.csv')
# create eqfit object
eq = eqfit.fitter() 
# sets the input parameters for the equation you want make.
# you can check these at any point by calling eq.X
eq.set_inputs(data.drop(columns=['Y']))
# sets the prediction column (i.e. what you want the equation to calculate)
# you can check this at any point by calling eq.Y
eq.set_target(data['Y'])
# splits the data in testing and training
eq.train_test_split()
# runs the polyfit which iteratively goes through different polynomial degrees and attempts to fit the data.
eq.do_polyfit(verbose=True)
# makes an equation from the best performing polynomial degree tested
# removes polynomial terms where the effective coefficient is below 100
eq.make_equation(param_notation=['A', 'B', 'C', 'D', 'E'], coef_cutoff=100)
# uses the created equation to calculate the equation predicted target values and compares them to the ones set by eq.set_target.
eq.evaluate_equation()
```

This covers the various functions included. For further details on each function please refer to the docstring by calling the help function. For example:
 ```
 help(eq.do_polyfit)
 ```

## Dependencies
- [Numpy](https://numpy.org/)
- [Sklearn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
