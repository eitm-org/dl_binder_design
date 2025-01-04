import os
from setuptools import setup, find_packages

setup(
      name='dl_binder_design',
      version='0.0.1',
      description='alphafold2 initial guess (but no proteinMPNN)',
      url = 'https://github.com/eitm-org/dl_binder_design',
      packages=find_packages(include=['mpnn_fr', 'af2_initial_guess', 'af2_initial_guess/alphafold', 'af2_initial_guess/alphafold/*', 'include' ]),
      install_requires=[
            'tqdm',
            'numpy',
            'torch',
            'scipy',
            'wandb',
            'tensorboard',
            'pytorch_lightning'
      ],
      include_package_data=True
)