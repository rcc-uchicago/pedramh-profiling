
Lmod is automatically replacing "intel/24.0" with "gcc/15.1.0".


Lmod is automatically replacing "impi/21.11" with "openmpi/5.0.8".


Lmod is automatically replacing "gcc/15.1.0" with "nvidia/25.3".


Lmod is automatically replacing "nvidia/25.3" with "opencilk/2.1.0".

Lmod has detected the following error: The following module(s) are unknown:
"conda"

Please check the spelling or version number. Also try "module spider ..."
It is also possible your cache file is out-of-date; it may help to try:
  $ module --ignore_cache load "conda"

Also make sure that all modulefiles written in TCL start with the string
#%Module

If this module depends on others you loaded, try loading prerequisites first,
then this module in a separate command.



/var/spool/slurmd/job3166613/slurm_script: line 33: syntax error near unexpected token `('
/var/spool/slurmd/job3166613/slurm_script: line 33: `export OMP_NUM_THREADS=max(cpus_per_gpu / num_data_workers, 1)'
