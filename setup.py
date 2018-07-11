from distutils.core import setup

setup(name='dm-utils',
      version='0.1',
      description='Utility packages for data mining.',
      url='',
      author='Ferenc Beres',
      author_email='fberes@info.ilab.sztaki.hu',
      license='SZTAKI DMS',
      packages=['dm_utils', 'dm_utils.model_wrappers', 'dm_utils.evaluation_utils'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
          'seaborn',
          'networkx',
          'sklearn',
          'xgboost',
          #'multiprocessing',
          #'functools',
          #'operator', 
          #'itertools'
      ],
      zip_safe=False)
