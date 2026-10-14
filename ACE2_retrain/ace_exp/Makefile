VERSION ?= $(shell git rev-parse --short HEAD)
IMAGE ?= fme
REGISTRY ?= registry.nersc.gov/m4492/ai2cm
ENVIRONMENT_NAME ?= fme
USERNAME ?= $(shell beaker account whoami --format=json | jq -r '.[0].name')
DEPLOY_TARGET ?= pypi
BEAKER_WORKSPACE = ai2/ace
NPROC ?= 2
FME_FORCE_CPU ?= 0
FME_DISTRIBUTED_BACKEND ?= torch
FME_DISTRIBUTED_H ?= 1
FME_DISTRIBUTED_W ?= 1
TEST_PATH ?= .

# Conda platform ("subdir") for this host. conda will happily install packages
# from the wrong subdir into an environment -- e.g. linux-64 (x86-64) binaries
# into an aarch64 env -- which produces libraries the loader silently rejects at
# runtime. Pinning the subdir at create time, and again in the env's own
# .condarc, prevents that. Override for an unlisted platform, e.g.
#   make create_environment CONDA_SUBDIR=linux-ppc64le
ifeq ($(shell uname -m), aarch64)
CONDA_SUBDIR ?= linux-aarch64
else
CONDA_SUBDIR ?= linux-64
endif

# conda-forge names its compiler toolchains after the target platform
# (gxx_linux-64, gxx_linux-aarch64, ...), so this must track the subdir. A
# mismatched toolchain still "installs", but its activate.d hooks reference an
# unset CONDA_BUILD_SYSROOT, which aborts activation under `set -u`.
ifeq ($(shell uname), Linux)
	CONDA_PACKAGES=gxx_$(CONDA_SUBDIR) pip
else
	CONDA_PACKAGES=pip
endif

build_docker_image:
	DOCKER_BUILDKIT=1 docker build --platform=linux/amd64 -f docker/Dockerfile -t $(IMAGE):$(VERSION) --target production .

push_shifter_image: build_docker_image
	docker tag $(IMAGE):$(VERSION) $(REGISTRY)/$(IMAGE):$(VERSION)
	docker push $(REGISTRY)/$(IMAGE):$(VERSION)

build_beaker_image: build_docker_image
	beaker image create --name $(IMAGE)-$(VERSION) $(IMAGE):$(VERSION)

build_podman_image:
	podman-hpc build -f docker/Dockerfile -t $(IMAGE):$(VERSION) .

migrate_podman_image: build_podman_image
	podman-hpc migrate $(IMAGE):$(VERSION)

enter_docker_image: build_docker_image
	docker run -it --rm $(IMAGE):$(VERSION) bash

launch_beaker_session:
	./launch-beaker-session.sh $(USERNAME)/$(IMAGE)-$(VERSION)

build_deps_only_image:
	DOCKER_BUILDKIT=1 docker build --platform=linux/amd64 -f docker/Dockerfile -t $(IMAGE)-deps-only:$(VERSION) --target deps-only .
	beaker image create $(IMAGE)-deps-only:$(VERSION) --name $(IMAGE)-deps-only-$(VERSION) --workspace ai2/ace-ci-tests

build_nsight_image:
	DOCKER_BUILDKIT=1 docker build --platform=linux/amd64 -f docker/Dockerfile -t $(IMAGE)-nsight:$(VERSION) --target nsight .
	beaker image create $(IMAGE)-nsight:$(VERSION) --name $(IMAGE)-nsight-$(VERSION) --workspace ai2/ace

# recommended to deactivate current conda environment before running this
create_environment:
	CONDA_SUBDIR=$(CONDA_SUBDIR) conda create -n $(ENVIRONMENT_NAME) python=3.11 $(CONDA_PACKAGES)
	conda config --file "$$(conda run --no-capture-output -n $(ENVIRONMENT_NAME) python -c 'import sys; print(sys.prefix)')/.condarc" --set subdir $(CONDA_SUBDIR)
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) python -m pip install uv
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) uv pip install -c constraints.txt -e .[dev,docs,graphcast]
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) uv pip install --no-build-isolation -c constraints.txt -r requirements-healpix.txt
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) uv pip install -r analysis-deps.txt

test:
	pytest -n 4 --durations 40 $(TEST_PATH)

test_parallel:
	FME_FORCE_CPU=$(FME_FORCE_CPU) \
	FME_DISTRIBUTED_BACKEND=$(FME_DISTRIBUTED_BACKEND) \
	FME_DISTRIBUTED_H=$(FME_DISTRIBUTED_H) \
	FME_DISTRIBUTED_W=$(FME_DISTRIBUTED_W) \
	torchrun --nproc-per-node $(NPROC) -m pytest -m parallel $(TEST_PATH)

# matrix for parallel tests
# | backend | WS | H | W |
# | ------- | -- | - | - |
# | torch   |  2 | 1 | 1 |
# | torch   |  3 | 1 | 1 |
# | model   |  3 | 1 | 1 |
# | model   |  2 | 2 | 1 |
# | model   |  2 | 1 | 2 |
# | model   |  3 | 3 | 1 |
# | model   |  3 | 1 | 3 |
# | model   |  4 | 2 | 2 |
# | model   |  4 | 2 | 1 |
# | model   |  8 | 2 | 2 |
cpu_test_all_parallel:
	FME_FORCE_CPU=1 NPROC=2 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=1 make test_parallel
	FME_FORCE_CPU=1 NPROC=2 FME_DISTRIBUTED_BACKEND=torch FME_DISTRIBUTED_H=1 FME_DISTRIBUTED_W=1 make test_parallel
	FME_FORCE_CPU=1 NPROC=3 FME_DISTRIBUTED_BACKEND=torch FME_DISTRIBUTED_H=1 FME_DISTRIBUTED_W=1 make test_parallel
	FME_FORCE_CPU=1 NPROC=3 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=1 FME_DISTRIBUTED_W=1 make test_parallel
	FME_FORCE_CPU=1 NPROC=2 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=1 FME_DISTRIBUTED_W=2 make test_parallel
	FME_FORCE_CPU=1 NPROC=4 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=2 make test_parallel
	FME_FORCE_CPU=1 NPROC=4 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=1 make test_parallel
	FME_FORCE_CPU=1 NPROC=8 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=2 make test_parallel

# --cov must come  after pytest args to use the sources defined by config
test_cov:
	pytest -n 4 --durations 40 --cov --cov-report=term-missing:skip-covered --cov-config=pyproject.toml $(TEST_PATH)

test_fast:
	pytest -n 4 --durations 40 --fast $(TEST_PATH)

test_very_fast:
	pytest --durations 40 --very-fast $(TEST_PATH)

# For maintainer use only
# requires fme[deploy] to be installed

build_pypi:
	rm -rf dist
	python -m build

deploy_pypi: build_pypi
	twine upload --repository $(DEPLOY_TARGET) dist/*

deploy_test_pypi: DEPLOY_TARGET = testpypi
deploy_test_pypi: deploy_pypi
