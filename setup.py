from setuptools import setup, find_packages

setup(
    name='advantitious-bush',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies here
    ],
    author='senseiwhales',
    author_email='senseiwhales@gmail.com',
    description='A Python library for implementing the Advantitious Bush algorithm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/senseiwhales/advantitious-bush',
    license='Apache License 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
