################
Dependencies
################
  
ZEN-garden is developed with Python version 3.11. and  depends on the following Python libraries:

* [**Pandas**](https://pandas.pydata.org/) for structuring and manipulating data about elements and their time series
* [**Numpy**](https://pandas.pydata.org/) for calculations
* [**Linopy**](https://pandas.pydata.org/) for constructing the optimization problem
* [**Pint**](https://pint.readthedocs.io/en/stable/) for  for handling physical quantity units efficiently
* [**Pytest**](https://docs.pytest.org/en/8.0.x/) for automatic testing
* [**Tsam**](https://tsam.readthedocs.io/en/latest/) for time series aggregation

To solve the optimization problem, the free solver `HiGHS <https://highs.dev/>`_ or the commercial solver `Gurobi <https://www.gurobi.com/>`, which offers free academic licenses, can be used.
