import os
from setuptools import setup, find_packages

# Get the directory where setup.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Read the contents of README.md
with open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='AdvantitiousBush',
    version='1.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies here
    ],
    author='senseiwhales',
    author_email='senseiwhales@gmail.com',
    description='A Python library for implementing the Advantitious Bush algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/senseiwhales/advantitious-bush',
    license='Apache License 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
