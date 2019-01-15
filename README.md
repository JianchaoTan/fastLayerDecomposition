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

    In a new terminal:
    cd image-layer-updating-GUI
    http-server

Some videos of GUI usage can be found in [this link](https://cragl.cs.gmu.edu/fastlayers/)

The `turquoise.png` image is copyright [Michelle Lee](https://cargocollective.com/michellelee).


## Image Recoloring GUI:

Users can perform global recoloring in our web GUI: https://yig.github.io/image-rgb-in-3D/
First load the original image, then drag-and-drop saved palette .js file, and finally drag-and-drop saved mixing weights .js file. Then user can click and move the palette vertices in GUI to do image recoloring.
This image recoloring web GUI is also used in our previous project: https://github.com/JianchaoTan/Decompose-Single-Image-Into-Layers


## Dependencies
* Python3.6
* NumPy
* SciPy
* Cython
* [GLPK](https://www.gnu.org/software/glpk/) (`brew install glpk`)
* cvxopt, built with the [GLPK](https://www.gnu.org/software/glpk/) linear programming solver interface (`CVXOPT_BUILD_GLPK=1 pip install cvxopt`)
* PIL or Pillow (Python Image Library) (`pip install Pillow`)
* pyopencl
* websockets (`pip install websockets`)
