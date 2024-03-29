from setuptools import setup, find_packages

contrib = [
    'Raphael Ortiz',
]

# setup.
setup(name='improc',
      version='0.2',
      description='Collection of image processing routines',
      author=', '.join(contrib),
      packages=find_packages(exclude=[
          'tests',
      ]),
      install_requires=[
          'numpy>=1.15.4', 'scikit-image', 'scipy', 'parse', 'vtk', 'czifile',
          'pandas'
      ],
      zip_safe=False)
