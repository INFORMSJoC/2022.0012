# subfun
This folder contains 31 subfunctions of the corresponding main functions in `mainfun`.
### Subfunctions called by the main function *NALM.m* include: 
  - *CholHess.m*:  Cholesky fractorization of Hessian
  - *Cumpute_matrix_C.m*: calculate a matrix C as a part of Hessian
  - *findstep*: find a step length by the line search
  - *Generatedash_Jacobi*: obtain some information of generated Jacobian matrix
  - *invDvec.m*: calculate D^{-1}*rhs and T defined in Section 3.3 of the paper
  - *linsysolvefun.m*: solve a linear system
  - *matvec_N_ALM.m*: compute a Hessian times a vector
  - *Proj_dknorm.m*: Compute the projection on the ball with center 0 and radius r in the sense of the dual norm of k-norm
### Subfunctions called by the main function *SIRPN_CD.m* include: 
  - *CD.m*: coordinate descent algorithm for solving subproblems of ***S-IRPN***.
  - *phi_eps_fun.m*: calculate the function value, first-order derivative, second-oeder derivative of phi defined in Appendix E of the supplementary materials
  - *psi_eps_fun.m*: calculate the function value, first-order derivative, second-oeder derivative of psi defined in Appendix E of the supplementary materials
### Subfunctions called by the main function *ADMM_knorm.m* include: 
  - *CholHess.m*:  Cholesky fractorization of Hessian
  - *linsysolvefun.m*: solve a linear system 
  - *Matvecmu.m*: compute a matrix times a vector when the sample size is not greater that the feature size
  - *Matvecnx.m*: compute a matrix times a vector when the sample size is greater that the feature size
  - *Proj_dknorm.m*: Compute the projection on the ball with center 0 and radius r in the sense of the dual norm of k-norm


### Subfunctions called by the main function *PSG_solver.m*include: 
  - 
  -
  -
### Subfunctions called by the main function *Gurobi_knorm_lp.m* include: 
  - *linprog.m*: solve LP using the Gurobi MATLAB interface
  -
  -
### Subfunctions called by the main function *AS_NALM_path.m* include: 
  - *Proj_dknorm.m*: Compute the projection on the ball with center 0 and radius r in the sense of the dual norm of k-norm
  -
  -
### Subfunctions called by the main function *MM_NPPA.m* include: 
  - 
  -
  -
### Subfunctions called by the main subfunction *dPPA_SSN.m* include: 
  - *CholHess.m*:  Cholesky fractorization of Hessian 
  - *Cumpute_matrix_W_MM.m*: calculate an element W in the Clarke generalized Jacobian of the proximal mapping with respect to k-norm
  - *findstep_new_MM.m*: find a step length by the line search
  - Generatedash_Jacobi_MM.m: obtain some information of generated Jacobian matrix
  - *invDvec_MM.m*: calculate D^{-1}*rhs and T 
  - *linsysolvefun.m*: solve a linear system
  - *Proj_dknorm.m*: Compute the projection on the ball with center 0 and radius r in the sense of the dual norm of k-norm
### Subfunctions called by the main function *MM_Gurobi.m* include: 
  - 
  -
  -
### Subfunctions called by the main subfunction *qp_Gurobi.m* include:
  - 
  -
  -
### Subfunctions called by the main function *Cross_validation_CVaR.m* include:  
  - 
  -
  -
### Subfunctions called by the main function *Cross_validation_truncated_CVaR.m* include:  
  - 
  -
  -
  
  
  
### functions for generating random data based on Table 1 or Table 3 in the supplementary materials
  - *Generate_A_b_t.m*: generate the small-scale random data based on Table 1 
  - *Generate_A_b_t_large.m*: generate the large-scale random data based on Table 1 
  - *Generate_A_b_Configuration.m*: generate the small-scale random data based on Table 3
  - *Generate_A_b_Conf_large.m*: generate the large-scale random data based on Table 3 
  - *Generate_Toeplitz_matrix.m*: generate n*n-dimensional Toeplitz covariance matrix
### Other subfunctions
  - *HKLeq.m*: subfunction of *Proj_dknorm.m* and *Proj_dknorm_wt.m*.
 
