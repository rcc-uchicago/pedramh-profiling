Lmod has detected the following error: The following module(s) are unknown:
"conda"

Please check the spelling or version number. Also try "module spider ..."
It is also possible your cache file is out-of-date; it may help to try:
  $ module --ignore_cache load "conda"

Also make sure that all modulefiles written in TCL start with the string
#%Module

If this module depends on others you loaded, try loading prerequisites first,
then this module in a separate command.



2026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu1242026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu1242026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu124
2026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu124

2026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu124

2026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu1242026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu124

2026-05-29 17:42:40,250 - root - INFO - Torch version: 2.6.0+cu124
Traceback (most recent call last):
Traceback (most recent call last):
  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
        params = YParams(os.path.abspath(args.yaml_config), args.config)params = YParams(os.path.abspath(args.yaml_config), args.config)

                          ^^^^^^Traceback (most recent call last):
^^^^^^^  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
^^^^^^^^^^^^^^^^^^^^^^Traceback (most recent call last):
^^^^^^  File "/work2/11095/jwan4/PanguWeather/v2.0/train.py", line 3546, in <module>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^params = YParams(os.path.abspath(args.yaml_config), args.config)^^


  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
            params = YParams(os.path.abspath(args.yaml_config), args.config) 
    ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^with open(yaml_filename) as _file:^^
^^^^ ^ ^^ ^    ^ ^with open(yaml_filename) as _file:^ ^
^ ^^ ^^  ^^  ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^^^^^^^^^^^^^^^^^^^^^
^^^^^^  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
^^^^^^^^^^^^^
^^^^IsADirectoryError^^: ^^[Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'^^
^^^^^
^
IsADirectoryError:   File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
[Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'
    with open(yaml_filename) as _file:
    with open(yaml_filename) as _file: 
              ^ ^ ^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^^IsADirectoryError^: 
[Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'
IsADirectoryError: [Errno 21] Is a directory: '/work2/11095/jwan4/PanguWeather/v2.0'
W0529 17:42:43.793000 3386516 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3386526 closing signal SIGTERM
E0529 17:42:43.825000 3386516 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 3386527) of binary: /work/11095/jwan4/conda-envs/sfno_pangu/bin/python
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
  time      : 2026-05-29_17:42:43
  host      : c562-005.stampede3.tacc.utexas.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 3386528)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2026-05-29_17:42:43
  host      : c562-005.stampede3.tacc.utexas.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 3386529)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-05-29_17:42:43
  host      : c562-005.stampede3.tacc.utexas.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 3386527)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
