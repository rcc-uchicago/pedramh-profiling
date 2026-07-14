
Lmod is automatically replacing "intel/24.0" with "gcc/15.1.0".


Lmod is automatically replacing "impi/21.11" with "openmpi/5.0.8".


Lmod is automatically replacing "gcc/15.1.0" with "nvidia/25.3".


Lmod is automatically replacing "nvidia/25.3" with "opencilk/2.1.0".

Traceback (most recent call last):
  File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <module>
Traceback (most recent call last):
  File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <module>
    params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]
    params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs] 
                      Traceback (most recent call last):
        File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <module>
  ^ ^ ^ ^ ^ ^^^^^^^^^^^^^Traceback (most recent call last):
^^^^^^  File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <module>
^^^^^^^    ^^params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]^^
^^^^^^^^^^ ^^ ^ ^^ ^^ ^^ ^^     ^^ params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]^^ 
^^ ^^ ^^ ^^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^^^ ^^^ ^^^ ^^^ ^^^ ^ ^^^ ^^^ ^^^ ^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^^^^^
^  File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <listcomp>
^^^  File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <listcomp>
^^^^^^^^^^^^^^^^^^^^^^^^^    ^^params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]^^    ^
^params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]^^
^^ ^^  ^^ ^ ^ ^ ^ ^ ^ ^^  ^^ ^ ^^  ^^  ^^  ^^  ^^  ^^  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^^^ 
^^ ^^^  File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <listcomp>
^^^^^^^^^^^^^^^^^^^^
^^^^  File "/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py", line 1195, in <listcomp>
^^^^    ^^params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]^^
^^^^^^ ^^ ^ ^^ ^^ ^    ^ ^params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]^ ^
^ ^^  ^^  ^^  ^^ ^ ^ ^ ^^  ^^  ^^  ^^  ^^  ^^  ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^^^^^^^^^^^^^^^^^^^
^^^^^^  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
^^
^^^^  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^^  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
^^^^^
  File "/work2/11095/jwan4/PanguWeather/v2.0/utils/YParams.py", line 16, in __init__
    with open(yaml_filename) as _file:
    with open(yaml_filename) as _file: 
     with open(yaml_filename) as _file: 
                     ^ with open(yaml_filename) as _file: ^ 
 ^  ^^  ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^^^^^^FileNotFoundError^^^: 

^[Errno 2] No such file or directory: '/work/11095/jwan4/PanguWeather/v2.0/config/stampede/E3SM_SFNO_H5_STAMPEDE_jsw.yaml'^
FileNotFoundErrorFileNotFoundError^: : ^[Errno 2] No such file or directory: '/work/11095/jwan4/PanguWeather/v2.0/config/stampede/E3SM_SFNO_H5_STAMPEDE_jsw.yaml'[Errno 2] No such file or directory: '/work/11095/jwan4/PanguWeather/v2.0/config/stampede/E3SM_SFNO_H5_STAMPEDE_jsw.yaml'^

^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/11095/jwan4/PanguWeather/v2.0/config/stampede/E3SM_SFNO_H5_STAMPEDE_jsw.yaml'
E0616 07:46:59.575000 1862134 /work2/11095/jwan4/conda-envs/sfno_pangu/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 1862203) of binary: /work/11095/jwan4/conda-envs/sfno_pangu/bin/python
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
/work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-06-16_07:46:59
  host      : c563-002.stampede3.tacc.utexas.edu
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 1862204)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2026-06-16_07:46:59
  host      : c563-002.stampede3.tacc.utexas.edu
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 1862205)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2026-06-16_07:46:59
  host      : c563-002.stampede3.tacc.utexas.edu
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 1862206)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-06-16_07:46:59
  host      : c563-002.stampede3.tacc.utexas.edu
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 1862203)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
