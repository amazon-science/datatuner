# Preparing MTurk Data

This directory contains scripts used for sampling the system generated outputs for human annotation, and the results of these annotations. 

## Generate fluency data for annotation

`python experiments/mturk/prepare_mturk.py prepare ~/system_outputs/ ~/mturk_fluency/ fluency`


## Generate fidelity data for annotation
`python experiments/mturk/prepare_mturk.py prepare ~/system_outputs/ ~/mturk_fidelity/ fidelity`


## Score the fluency annotations
`python experiments/mturk/prepare_mturk.py score ~/system_outputs_test/ ~/mturk_fluency/  fluency ./experiments/mturk/`

## Score the fidelity annotations
`python experiments/mturk/prepare_mturk.py score ~/system_outputs_test/ ~/mturk_fidelity/  fidelity ./experiments/mturk/`