This repository contains code for the Energy 291 final project.

## Installation ##
This repository uses cvxpy for the construction of the optimization problems and PyTorch for training the neural network models. The necessary Python packages can be instaled by running the following command:

```
python3 -m venv energy291_project

source energy291_project/bin/activate
deactivate

which python
pip install numpy ipython ipykernel
python -m ipykernel install --user --name energy291_project

pip install -r requirements.txt
```

## Necessary Data and Software ##
* We use the [Gurobi](https://support.gurobi.com/hc/en-us/community/posts/360046430451/comments/360005981732) commercial solver for solving the problems.
* Data for this was provided by the [California ISO](https://www.caiso.com/todaysoutlook/Pages/supply.html).
