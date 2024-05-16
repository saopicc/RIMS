from setuptools import setup, find_packages

setup(
    name='DynSpecMS',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # list of packages your project depends on
    ],
    entry_points={
        'console_scripts': [
            # allows command line execution of your functions
        ],
    },
    author='Cyril Tasse',
    author_email='cyril.tasse@obspm.fr',
    description='Extract Dynamic Spectra from Measurement Sets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cyriltasse/DynSpecMS',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)