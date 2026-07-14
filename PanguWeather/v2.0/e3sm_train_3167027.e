Lmod has detected the following error: These module(s) or extension(s) exist
but cannot be loaded as requested: "cuda/12.8"
   Try: "module spider cuda/12.8" to see how to load the module(s).
   The requested module(s) require a toolchain that is incompatible with the
currently loaded environment.



2026-05-29 22:37:04,543 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 22:37:04,543 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 22:37:04,544 - root - INFO - Torch version: 2.6.0+cu1242026-05-29 22:37:04,543 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 22:37:04,544 - root - INFO - Torch version: 2.6.0+cu124

2026-05-29 22:37:04,543 - root - INFO - Torch version: 2.6.0+cu124
Traceback (most recent call last):
  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
    params = YParams(os.path.abspath(args.yaml_config), args.config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
2026-05-29 22:37:04,547 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 22:37:04,547 - root - INFO - Torch version: 2.6.0+cu124
Traceback (most recent call last):
Traceback (most recent call last):
  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
    with open(yaml_filename) as _file:
         ^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'
        params = YParams(os.path.abspath(args.yaml_config), args.config)params = YParams(os.path.abspath(args.yaml_config), args.config)

                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
    with open(yaml_filename) as _file:
     with open(yaml_filename) as _file: 
            ^ ^ ^ ^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^^IsADirectoryError^: ^[Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'

IsADirectoryError: [Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'
Traceback (most recent call last):
  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
    params = YParams(os.path.abspath(args.yaml_config), args.config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
    with open(yaml_filename) as _file:
         ^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'
W0529 22:37:10.512000 4020045 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4020056 closing signal SIGTERM
E0529 22:37:10.544000 4020045 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 4020055) of binary: /work/11095/jwan4/conda-envs/sfno_pangu/bin/python
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
[1]:
  time      : 2026-05-29_22:37:10
  host      : c561-010.stampede3.tacc.utexas.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 4020057)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2026-05-29_22:37:10
  host      : c561-010.stampede3.tacc.utexas.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 4020058)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-05-29_22:37:10
  host      : c561-010.stampede3.tacc.utexas.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 4020055)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
