from setuptools import setup, find_packages

setup(
    name='DynSpecMS',
    version='0.1.0',
    packages=find_packages(include=['DynSpecMS', 'DynSpecMS.*']),
    include_package_data=True,
    install_requires=[
        'dask[array]',
        'dask-ms',
        'xarray',
        'psutil',
        'numpy',
        'matplotlib',
        'astropy',
        'future',
        'scipy',
        'jax',
        'jaxlib',
        'pydantic>=2.0.0',
        'requests',
        'kronicle_sdk'
        # other dependencies
    ],
    entry_points={
        'console_scripts': [
            'rims_run=DynSpecMS.scripts.ms2dynspec:main',
            'rims_publish=DynSpecMS.scripts.dynspec_upload:main' 
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