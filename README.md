[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Convex and Nonconvex Risk-based Linear Regression at Scale

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [General Public License v2.0](LICENSE).

The software in this repository is a snapshot of the software that was used in the research reported 
on in the paper [Convex and Nonconvex Risk-based Linear Regression at Scale](https://www.doi.org) by Can Wu, Ying Cui, Donghui Li and Defeng Sun.
The snapshot corresponds to [this release](https://github.com/Wu-Can/k-normCode) in the development repository. 

## Cite

To cite this material, please cite this repository, using the following DOI.

[![DOI](tbd)](tbd)

Below is the BibTex for citing this version of the code.

```
@article{Risk202Xregression,
  author =        {Can Wu, Ying Cui, Donghui Li and Defeng Sun},
  publisher =     {INFORMS Journal on Computing},
  title =         {Convex and Nonconvex Risk-based Linear Regression at Scale},
  year =          {202X},
  doi =           {TBD},
  url =          {https://github.com/INFORMSJoC/2022.0012},
}  
```

## Description

The goal of this software is to solve high-dimensional sparse linear regression problems 
under either the VaR or the CVaR risk measures. it is written in MATLAB and compares with the publically available [Gurobi optimizer](https://www.gurobi.com/downloads/gurobi-software/) and the [PSG program](http://www.aorda.com/index.php/downloading/).  For all of them,
free academic licenses are available.

####  Three types of optimization problems and the corresponding solvers

- CVaR regression with a fixed lambda: the convex CVaR-based sparse linear regression with a fixed value of lambda.
  - **N-ALM**: the semismooth Newton based on the proximal augmented Lagrangian method
  - **ADMM**: the alternating direction method of multipliers 
  - **Gurobi**: the barrier method in Gurobi
  - **S-IRPN**: the smoothing method based on the inexact regularized proximal Newton method 
  - **PSG solvers**: seven solvers incuding **VAN**, **TANK**, **CAR**, **BULDOZER**, **VANGRB**, **CARGRB** and **HELI** in the PSG package  
- CVaR regression with a sequence of lambda: the convex CVaR-based sparse linear regression with a given sequence of grid points lambda.
  - **AS+N-ALM**: the adaptive sieving strategy combined with **N-ALM**
  - **Warm+N-ALM**: the warm-strated **N-ALM**
  - **N-ALM**: the pure **N-ALM** 
- Truncated CVaR regression: the nonconvex truncated CVaR-based sparse linear regression.
  - **MM+N-PPA**: the majorization-minimization algorithm combined with the semismooth Newton method based on the proximal point algorithm
  - **MM+Gurobi**: the majorization-minimization algorithm combined with the barrier method in Gurobi

#### Two kinds of data sets

- Real data: eleven real data sets in LIBSVM format are all available on [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html).

- Random data: generated randomly under three different heavy-tailed
errors according to Table 1 or three different contamination schemes according to Table 3 in the supplementary materials.

#### Ten folders in the software

- `Test1_fixed_lambda`: test the performance of **N-ALM**, **ADMM**, **Gurobi**, **S-IRPN** and **PSG solvers** for CVaR regression with a fixed lambda.
- `Test2_Solution_path`: test the performance of **AS+N-ALM**, **Warm+N-ALM** and **N-ALM** for CVaR regression with a sequence of lambda.
- `Test3_truncated_CVaR_MM`: test the performance of **MM+N-PPA** and **MM+Gurobi** for truncated CVaR regression.
- `Test4_Risk-averse_model`: test the performance of the risk-sensitive and risk-neutral regression models.
- `genUCIdatafun`: convert the UCI data from LIBSVM format to MAT format.
- `mainfun`: all main functions for the above solvers.
- `mexfun`: two funtions in MEX format.
- `solver`: all the subfunctions called by the above main functions.
- `UCIdata`: all availiable data for the real data and the high-accurate objective values.
- `Results`: numerical results and their corresponding running records of Table 2 in the paper.

## Usage

All numerical results in the paper and supplementary materials were generated by this software that had been carried out using MATLAB R2020b on a desktop computer (16-core, Intel(R) Core(TM) i7-10700 @ 2.90GHz, 32G RAM). Please follow the steps below to obtain the corresponding results:

***Step 1***. Unpack the software and run Matlab in its directory.

***Step 2***. In the MATLAB command window, type:

```
    >> startup 
```

***Step 3***. Generate the expanded UCI data sets as follows:

  - Download and unpack eleven real data sets: 
  [abalone_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale), 
  [bodyfat_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat_scale),
  [housing_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale),
  [mpg_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg_scale), 
  [pyrim_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/pyrim_scale),
  [space_ga_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga_scale),
  [triazines_scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/triazines_scale),
  [E2006.test](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.test.bz2), [E2006.train](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2), [log1p.E2006.train](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.train.bz2) and [log1p.E2006.test](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.test.bz2);
  - Create a new folder `UCIdataorg` in `genUCIdatafun` and then put all eleven real data sets into the new folder;
  - Make the function *libsvmread* available: 
    - download and unpack the [LIBSVM package](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) in `genUCIdatafun`;
    - run *make.m* in `\genUCIdatafun\libsvm-3.3\matlab`;
  - Run *genUCIdata.m* in `\genUCIdatafun`, then the expanded UCI data sets in MAT format will be generated in the folder `UCIdata`. 

***Step 4***. Run the scripts for generating the corresponding results on UCI data in the paper according the following table:

<table>
	<tr>
		<th> Results</th>
		<th> Folders</th>
		<th> Scripts</th>
		<th> INPUT in Scripts</th>
	</tr>
	<tr>
		<td rowspan="3">Table 2</td>
		<td rowspan="3">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_UCI</td>
		<td>prob=[1:11]; flag_tol=1; tol=1e-3; alpha_vec=[0.9, 0.5, 0.1];</td>
	</tr>
	<tr>
		<td><i>Test_SIRPN_UCI</td>
		<td>prob=[1:11];</td>
	</tr>
	<tr>
		<td><i>Test_ADMM_UCI</td>
		<td>prob=[1:11]; flag_tol=1;</td>
	</tr>
	<tr>
		<td rowspan="2">Table 3</td>
		<td rowspan="2">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_UCI</td>
		<td>prob=[7:9]; flag_tol=0; tol=1e-6; alpha_vec=[0.9, 0.5, 0.1];</td>
	</tr>
	<tr>
	        <td><i>Test_PSG_UCI</td>
		<td>prob=[7:9]; SOLVER='VAN';</td>
	</tr>
	<tr>
		<td rowspan="2">Table 4</td>
		<td rowspan="2">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_UCI</td>
		<td>prob=[1:11]; flag_tol=0; tol=1e-8; alpha_vec=[0.9, 0.5, 0.1];</td>
	</tr>
	<tr>
	        <td><i>Test_Barrier_Gurobi_UCI</td>
		<td>prob=[1:11];</td>
	</tr>
	<tr>
		<td rowspan="3">Table 5</td>
		<td rowspan="3">`Test2_Solution_path`</td>
		<td><i>Test_AS_NALM_path_UCI</td>
		<td>prob=[9, 4, 2];</td>
	</tr>
	<tr>
		<td><i>Test_warm_NALM_path_UCI</td>
		<td>prob=[9, 4, 2];</td>
	</tr>
	<tr>
		<td><i>Test_NALM_path_UCI</td>
		<td>prob=[9, 4, 2];</td>
	</tr>
	<tr>
		<td>Figures 1 & 2</td>
		<td>`Test2_Solution_path/ Result_solution_path_figure_UCI`</td>
		<td><i>Test_path_figure_UCI</td>
		<td>flag_time=1; flag_nnz=1;</td>
	</tr>
	<tr>
		<td rowspan="2">Table 6</td>
		<td rowspan="2">`Test3_truncated_CVaR_MM`</td>
		<td><i>Test_MM_NPPA_UCI</td>
		<td>prob=[10 11 5 8 9];</td>
	</tr>
	<tr>
	        <td><i>Test_MM_Gurobi_UCI</td>
		<td>prob=[10 11 5 8 9];</td>
</table>

***Step 5***. Run the scripts for generating the corresponding results on random data in the supplementary materials according the following table:

<table>
	<tr>
		<th> Results</th>
		<th> Folders</th>
		<th> Scripts</th>
		<th> INPUT in Scripts</th>
	</tr>
	<tr>
		<td>Table 2</td>
		<td>`Test4_Risk-averse_model`</td>
		<td><i>Test_convex_CVaR_model_random</td>
		<td>flag_err=1 or 2 or 3;</td>
	</tr>
	<tr>
		<td>Table 4</td>
		<td>`Test4_Risk-averse_model`</td>
		<td><i>Test_truncated_CVaR_model_random</td>
		<td>conf=1;</td>
	</tr>
	<tr>
		<td>Table 5</td>
		<td>`Test4_Risk-averse_model`</td>
		<td><i>Test_truncated_CVaR_model_random</td>
		<td>conf=2;</td>
	<tr>
		<td rowspan="3">Table 6</td>
		<td rowspan="3">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_random</td>
		<td>prob=[5:7]; flag_tol=1; tol=1e-3; alpha_vec=[0.9, 0.5, 0.1]; flag_J=0;</td>
	</tr>
	<tr>
		<td><i>Test_SIRPN_random</td>
		<td>prob=[5:7];</td>
	</tr>
	<tr>
		<td><i>Test_ADMM_random</td>
		<td>prob=[5:7]; flag_tol=1;</td>
	</tr>
	<tr>
		<td rowspan="2">Table 7</td>
		<td rowspan="2">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_random</td>
		<td>prob=[1 2]; flag_tol=0; tol=1e-6; alpha_vec=[0.9, 0.5, 0.1]; flag_J=0;</td>
	</tr>
	<tr>
	        <td><i>Test_PSG_random</td>
		<td>prob=[1 2]; solvers={all seven solvers};</td>
	</tr>
	<tr>
		<td rowspan="2">Table 8</td>
		<td rowspan="2">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_random</td>
		<td>prob=[3 4]; flag_tol=0; tol=1e-6; alpha_vec=[0.9, 0.5, 0.1]; flag_J=0;</td>
	</tr>
	<tr>
	        <td><i>Test_PSG_random</td>
		<td>prob=[3 4]; solvers={'VAN'};</td>
	</tr>
	<tr>
		<td rowspan="2">Table 9</td>
		<td rowspan="2">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_random</td>
		<td>prob=[5:7]; flag_tol=0; tol=1e-8; alpha_vec=[0.9, 0.5, 0.1]; flag_J=0;</td>
	</tr>
	<tr>
	        <td><i>Test_Barrier_Gurobi_random</td>
		<td>m_vec=[3e2, 1e3, 3e3];<br> n_vec=[5e4, 1e5, 5e5];</td>
	</tr>
	<tr>
		<td>Table 10</td>
		<td>`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_UCI</td>
		<td>prob=6; flag_tol=0; tol=1e-6; alpha_vec=1-[1e-3.*[1 5], 1e-2.*[1:2:9, 10:10:90]];</td>
	</tr>
	<tr>
		<td>Table 11</td>
		<td>`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_random</td>
		<td>prob=7; flag_tol=0; tol=1e-6; alpha_vec=1-[1e-3.*[1 5], 1e-2.*[1:2:9, 10:10:90]]; flag_J=0;</td>
		</tr>
	<tr>
		<td rowspan="2">Figure 1</td>
		<td rowspan="2">`Test1_fixed_lambda`</td>
		<td><i>Test_NALM_random</td>
		<td>prob=7; flag_tol=0; tol=1e-6; alpha_vec=[0.4:-0.1:0.1]; flag_J=1;</td>
	</tr>
	<tr>
	        <td><i>Figure_indexJ</td>
		<td>iter_0=6; iter_1=20; m=3e3; n=5e5;</td>
	</tr>
	<tr>
		<td rowspan="3">Table 12</td>
		<td rowspan="3">`Test2_Solution_path`</td>
		<td><i>Test_AS_NALM_path_random</td>
		<td rowspan="3">m_vec=1e3.*[0.5, 1, 3]; n_vec=1e5.*[1, 2, 5]; edgp_mat=[15 30 45; 15 30 45; 30 60 90];</td>
	</tr>
	<tr>
		<td><i>Test_warm_NALM_path_random</td>
	</tr>
	<tr>
		<td><i>Test_NALM_path_random</td>
	</tr>
	<tr>
		<td>Figures 2 & 3</td>
		<td>`Test2_Solution_path/ Result_solution_path_figure_random`</td>
		<td><i>Test_path_figure_random</td>
		<td>flag_time=1; flag_nnz=1;</td>
	</tr>
	<tr>
		<td rowspan="2">Table 13</td>
		<td rowspan="2">`Test3_truncated_CVaR_MM`</td>
		<td><i>Test_MM_NPPA_random</td>
		<td rowspan="2">m_vec=1e3.*[0.5, 0.5, 1, 1.5, 3]; <br>n_vec=1e5.*[0.03, 0.5, 1, 2, 5]; lamc_vec=[0.15 0.1 0.05];</td>
	</tr>
	<tr>
	        <td><i>Test_MM_Gurobi_random</td>
	
</table>

	
## Example
If you want to see the performance of **N-ALM**, **S-IRPN** and **ADMM** on UCI data with epsilon=1e-3 in Table 2 of the paper, you only need to execute the following three operations in `Test1_fixed_lambda` according to ***Step 4***:
- Modify the INPUT part of the script *Test_NALM_UCI.m* to
	
  ```
    prob=[1:11]; flag_tol=1; tol=1e-3; alpha_vec=[0.9, 0.5, 0.1];
  ```
	then run this script.
- Modify the INPUT part of the script *Test_SIRPN_UCI.m* to	
	
  ```
    prob=[1:11]; 
  ```
	then run this script.
- Modify the INPUT part of the script *Test_ADMM_UCI.m* to	
	
  ```
    prob=[1:11]; flag_tol=1;
  ```
	then run this script.
	
During the above three operations, you will see the information for each iteration of **N-ALM**, **S-IRPN** and **ADMM** in the current command window (see Diary_NALM_UCI_flagtol_1_tol_1e-03.txt, Diary_SIRPN_UCI.txt and Diary_ADMM_UCI.txt in `Results` folder), respectively. Finally, you will obtain the files Result_NALM_UCI_flagtol_1_tol_1e-03.mat, Result_SIRPN_UCI.mat and Result_ADMM_UCI.mat in the current folder (see these three files in`Results` folder), which include all the information required in Table 2.

## Replicating
- To replicate all the results on UCI data in the paper, modify and run the scripts in the corresponding folders `Test1_fixed_lambda`, `Test2_Solution_path` and `Test3_truncated_CVaR_MM` according to ***Step 4***, respectively. 
- To replicate all the results on random data in the supplementary materials, modify and run the scripts in the corresponding folders `Test1_fixed_lambda`, `Test2_Solution_path`,  `Test3_truncated_CVaR_MM` and `Test4_Risk-averse_model` according to ***Step 5***, respectively. 
  
## Remark
As mentioned in our paper, once two algorithms are stopped under different criteria, we use the objective values obtained from **Gurobi** or **N-ALM** under the tolerance 1e-9 as benchmarks to test the quality of the computed objective values by both algorithms. For convenience, the corresponding high-accurate objective values are available in the `UCIdata` folder.











