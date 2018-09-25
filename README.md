# Efficient palette-based decomposition and recoloring of images via RGBXY-space geometry


This code implements the pipeline described in the SIGGRAPH Asia 2018 paper ["Efficient palette-based decomposition and recoloring of images via RGBXY-space geometry"](https://cragl.cs.gmu.edu/fastlayers/) Jianchao Tan, Jose Echevarria, and Yotam Gingold.

A different and simpler prototype implementation can be found in [this link](https://cragl.cs.gmu.edu/fastlayers/RGBXY_weights.py)

## Example usage

Run whole pipeline:

    Directly run `Our_preprocessing_pipeline.ipynb` using jupyter notebook

You can test if your installation is working by comparing your output to the `test/turquoise groundtruth results/` directory.

Launch GUI:

    cd image-layer-updating-GUI
    python3 server.py
    http-server


The `turquoise.png` image is copyright [Michelle Lee](http://cargocollective.com/michellelee/Illustration).

## Dependencies
* NumPy
* SciPy
* Cython
* [GLPK](https://www.gnu.org/software/glpk/) (`brew install glpk`)
* cvxopt, built with the [GLPK](https://www.gnu.org/software/glpk/) linear programming solver interface (`CVXOPT_BUILD_GLPK=1 pip install cvxopt`)
* PIL or Pillow (Python Image Library) (`pip install Pillow`)
* pyopencl
* websockets (`pip install websockets`)
