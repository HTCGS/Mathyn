# Mathyn

Mathyn is an educational Python project with numerical tools for computation
and mathematical exploration. The library includes numerical differentiation
and integration, linear algebra and vector calculus operations, and coordinate
transformations.

The project is under development, and its API and module organization may
change. Numerical results depend on the selected step size and number of
subintervals; verify the accuracy for your use case.

## Project structure

- `mathyn.differentiation` — numerical derivatives, partial derivatives, and
  gradients.
- `mathyn.integration` — numerical quadrature methods, multiple integrals, and
  related calculations.
- `mathyn.linear_algebra` — vector operations, including dot and cross
  products, magnitude, projection, angles, and direction cosines.
- `mathyn.vector_calculus` — tangent and normal objects, vector functions, and
  vector calculus integrals.
- `mathyn.coordinate_transforms` — transformations to polar, cylindrical, and
  spherical coordinates.

The repository root also contains `main.py`, an example script, and
`math.ipynb`, a Jupyter Notebook.

## Requirements

The `environment.yml` file contains a Conda environment configuration using
Python 3.10. When installing with `pip`, use a Python version compatible with
the packages in `requirements.txt`.

The core mathematical functions use NumPy. See `requirements.txt` for the
project's complete dependency list.

## Installation

Clone the repository and navigate to its root directory:

```bash
git clone https://github.com/HTCGS/Mathyn.git
cd Mathyn
```

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

## Example

Run Python from the repository root so that the `mathyn` package is available
for import:

```python
import mathyn


def f(x):
    return x**2


# derivative returns a function that computes the numerical derivative.
df = mathyn.differentiation.derivative(f)
print(df(3))  # approximately 6

# Integrate the function over an interval using Simpson's rule.
area = mathyn.integration.simpsons_rule(f, 0, 1, n=10)
print(area)  # approximately 1/3
```

## Run the example

From the project root, run:

```bash
python main.py
```

To use the notebook, open `math.ipynb` in Jupyter or VS Code and select a
Python interpreter with the dependencies from `requirements.txt` installed.
