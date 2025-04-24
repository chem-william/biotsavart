[![DOI](https://zenodo.org/badge/195408070.svg)](https://doi.org/10.5281/zenodo.14778252)

This is an Python implementation of the Biot-Savart law. It efficiently calculates the magnetic field of a 3D current density through a molecule.

It works in conjunction with the output from Python script [calc_current](https://github.com/chem-william/calc_current), but can reasonably easily be extended to use current density from any arbitrary calculation.

## Example

```bash
./biot.py --input ./path_to_current_c_all.npy --output path_to_visualization_output.spt
```

Feel free to message me with any questions.
