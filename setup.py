from distutils.core import setup

setup(name='dm-utils',
      version='0.1',
      description='Utility packages for data mining.',
      url='',
      author='Ferenc Beres',
      author_email='fberes@info.ilab.sztaki.hu',
      license='SZTAKI DMS',
      packages=['dm_utils', 'dm_utils.model_wrappers'],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'networkx',
          'sklearn',
          'xgboost'
      ],
      zip_safe=False)
