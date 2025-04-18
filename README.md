# <span style="color:#8a1dcf">C<sup>3</sup> : Climate - Code - Computation</span>

[![GitHub](https://img.shields.io/github/license/sebastian-mutz/c3)](https://github.com/sebastian-mutz/c3/blob/main/LICENCE)

A collection of teaching resources for computational climatology, including coding exercises (Fortran and Python ) organised into jupyter notebooks.

## <span style="color:#8a1dcf">Contents:</span>

### <span style="color:#4d4d4d">1. Exercises</span>

#### <span style="color:#4d4d4d">1.1 Energy Balance Model</span>

- **Description**: A simple 1D energy balance model that lets you calculate Earth's mean global temperature.
- **Notebook Implementations**: Fortran and Python

#### <span style="color:#4d4d4d">1.2 Wildfire Model (Euler Method) </span>

- **Description**: A simple 2D model that relies on the Euler method to simulate wildfire spread.
- **Notebook Implementations**: Fortran and Python

#### <span style="color:#4d4d4d">1.3 Wildfire Model (Cellular Automaton) </span>

- **Description**: A simple cellular automaton model to simulate wildfire spread.
- **Notebook Implementations**: Fortran and Python

#### <span style="color:#4d4d4d">1.4 Moutain Wave Model </span>

- **Description**: A simple model for prediction mountain waves and associated precipitation.
- **Notebook Implementations**: Fortran and Python

### <span style="color:#4d4d4d">2. Supplemental Code</span>

A separate .py and .f90 is provided with each notebook. These contain the code decoupled from the notebook.

## <span style="color:#8a1dcf">Progress:</span>

| Contents                   | Implemented |
| -------------------------- | ----------- |
| Energy Balance Model (f90) | -           |
| Energy Balance Model (py)  | ✓           |
| Wildfire Euler (f90)       | -           |
| Wildfire Euler (py)        | ✓           |
| Wildfire CA (f90)          | -           |
| Wildfire CA (py)           | -           |
| Moutain Wave Model (f90)   | -           |
| Moutain Wave Model (py)    | -           |


## <span style="color:#8a1dcf">Requirements:</span>

### Python:
- Numpy
- Math
- Matplotlib

### Fortran:
- [LFortran (interactive compiler)](https://github.com/lfortran/lfortran)
- [FPLT (Fortran Plotting Library))](https://github.com/sebastian-mutz/fplt)
