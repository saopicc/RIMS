from setuptools import setup, find_packages

setup(
    name='DynSpecMS',
    version='0.1.0',
    packages=find_packages(include=['DynSpecMS', 'DynSpecMS.*']),
    include_package_data=True,
    install_requires=[
        'dask[array]<=2023.5.0',  # Add this line
        'psutil<=5.9.3'
        # other dependencies
    ],
    entry_points={
        'console_scripts': [
            'ms2dynspec=DynSpecMS.scripts.ms2dynspec:main',  # Correct this line
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