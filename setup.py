from setuptools import setup, find_packages

setup(
    name="geopredictors",
    version="0.1",
    packages=find_packages(),
    description="A package for geo prediction. Contains Lithofacies classifier, permeability predictor",
    author="GEMS",
    author_email="yc4923@ic.ac.uk",
    install_requires= [line.rstrip(' \n') for line in open('requirements.txt')],
    include_package_data=True
)
