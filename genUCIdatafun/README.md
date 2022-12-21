# QCQP
This folder contains the folders and files to generate the results shown in Section 5.3 of the paper.
- `QCQP/test_QCQP.m`: the main script.
- `QCQP/algorithms`: contains the codes of **YNW**, **PSG**, **APriD** and **SLPMM** which are adjusted for stochastic quadratically constrained quadratic programming.
- `QCQP/results`: contains the generated figures.
- `QCQP/tools`: contains the necessary support scripts.

Note that `QCQP` does not contain a *data* folder, since  the tested numerical instances in this experiment are randomly generated in the main script by MATLAB functions. See a detailed description at Section 5.3 in the paper on how to randomly generate  the tested numerical instances.
