#!/bin/bash
export ONEAPI_DEVICE_SELECTOR=ext_oneapi_cuda:${SLURM_LOCALID}
exec "$@"