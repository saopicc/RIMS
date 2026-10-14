from setuptools import setup, find_packages

setup(
    name='DynSpecMS',
    version='0.1.0',
    packages=find_packages(include=['DynSpecMS', 'DynSpecMS.*']),
    include_package_data=True,
    install_requires=[
        'python-casacore',
        'dask[array]',
        'xarray',
        'zarr>=3',
        'psutil',
        'numpy',
        'matplotlib',
        'astropy',
        'future',
        'scipy',
        'jax',
        'jaxlib',
        'pydantic>=2.0.0',
        'pydantic[email]',
        'requests',
        'kronicle_sdk'
        # other dependencies
    ],
    extras_require={
        'profiling': ['nvtx'],  # NVTX ranges for Nsight Systems, see profiling/run_nsys.sh
    },
        entry_points={
        'console_scripts': [
            'rims=DynSpecMS.scripts.cli:main',
        ],
    },
    author='Cyril Tasse and the RIMS team',
    author_email='cyril.tasse@obspm.fr',
    description='Extract Dynamic Spectra from Measurement Sets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/saopicc/RIMS',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)