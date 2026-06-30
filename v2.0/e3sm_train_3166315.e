
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



2026-05-29 12:16:55,793 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 12:16:55,793 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 12:16:55,793 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 12:16:55,793 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 12:16:55,794 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 12:16:55,794 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 12:16:55,794 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 12:16:55,794 - root - INFO - Torch version: 2.6.0+cu124
[rank0]: Traceback (most recent call last):
[rank0]:   File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3795, in <module>
[rank0]:     os.makedirs(params['experiment_dir'], exist_ok=True)
[rank0]:   File "<frozen os>", line 215, in makedirs
[rank0]:   File "<frozen os>", line 215, in makedirs
[rank0]:   File "<frozen os>", line 215, in makedirs
[rank0]:   [Previous line repeated 5 more times]
[rank0]:   File "<frozen os>", line 225, in makedirs
[rank0]: PermissionError: [Errno 13] Permission denied: '/glade'
W0529 12:17:04.017000 4188352 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4188355 closing signal SIGTERM
W0529 12:17:04.018000 4188352 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4188356 closing signal SIGTERM
W0529 12:17:04.018000 4188352 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4188357 closing signal SIGTERM
E0529 12:17:04.546000 4188352 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 4188354) of binary: /work/11095/jwan4/conda-envs/sfno_pangu/bin/python
Traceback (most recent call last):
  File "/work/11095/jwan4/conda-envs/sfno_pangu/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-05-29_12:17:04
  host      : c561-007.stampede3.tacc.utexas.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 4188354)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
