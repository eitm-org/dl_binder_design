import os
from setuptools import setup, find_packages

setup(
      name='dl_binder_design',
      version='0.0.1',
      description='alphafold2 initial guess (but no proteinMPNN)',
      url = 'https://github.com/eitm-org/dl_binder_design',
      packages=find_packages(),
      install_requires=[
            'tqdm',
            'numpy',
            'torch',
            'scipy',
            'wandb',
            'tensorboard',
            'pytorch_lightning'
      ],
      package_data={'': ['mpnn_fr/RosettaFastRelaxUtil.xml']}
)