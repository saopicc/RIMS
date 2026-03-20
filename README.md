# RIMS
A Radio Astronomy Software tool for creating the _Dynamic Spectra_ for a specified point in the sky from image visibilities in a [Measurement Set](https://casaguides.nrao.edu/index.php?title=Measurement_Set_Contents) (MS). The tool offers algorithmic support for _beam correction_ and _re-phasing_ to provide an accurate _Jy/beam_ measure of the specified target. With little overhead, the tool also supports the generation of _Dynamic Spectra_ for multiple targets or off-targets. A larger sample of off-target (randomly sampled within a specified radius of the _Phase Center_) _Dynamic Spectra_  provide an increasingly accurate variance estimate for the target Dynamic Spectra when used with [Inspect_Dynspec](https://github.com/ratt-ru/inspect-dynspec).

### Installation
#### Requirements:
- Ubuntu 20.04 or 22.04
- <= Python3.12 virtual environment
- DDFacet installed to virtual environment
- Casa5 installation

#### Steps:
```
git clone https://github.com/saopicc/RIMS.git
pip install ./RIMS
```

### RIMS run
For a given Measurement set and list of targets you're interested in, generate dynamic spectra for every individual target.

### Usage
View help:
```
rims run --help
```
Run job:
```
rims run --ms <ms location> --data <predicted visibility column> --model <model visibilty column> --srclist <src list location> --rad <radius from which to sample off targets> --noff <number of off targets> --DDFParset <path to parset file for beam correction> --CacheDir <temp directory to write cache> --OutDirName <output directory name>
```
Notes:

- If you want to creat _Dynamic Spectra_ from residual visibilities, you can specify the model (--model) column and the tool with automatically subtract the model from the data (--data) column.
- Your target in the source list (--srclist) must be within the radius (--rad) specified from _Phase Center_. 

### Creating a source file:
A source file may either be in [ECSV](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) format or a simple `csv` format of the following structure:
```
# Example source list
Name ra dec
Deneb    20:41:25.9   +45:16:49
etc      12:00:00.0   +42:00:00
```

or in fits table format

### Output:
The output folder should countain the following files/folders:
```
├── <dataset-id>.reg
├── Catalog.npy
├── run_metadata.json
├── OFF
├── OFF_W
├── TARGET
└── TARGET_W
```
Where the OFF and OFF_W directories contain the off target and off target weights fits files respectively, and the TARGET and TARGET_W directories contain the target and target weights fits files respectively.

### RIMS publish
Should you wish to contribute to the global "RIMS online" project, we encourage you to upload your generated dynamic spectra. To do so, 

### Usage
View help:
```
rims publish --help
```
Run job:
```
rims publish <path to 'rims run' output> ---server-conf <path to server.ini> --publisher-conf <path to details.ini>
```

Where the contents of the server conf file looks like:
```
[upload]
host=<url for server host>
token=<upload token for user for server host>

[kronicle]
host=https://kronicle.aqmo.org
username=<kronicle provided username>
password=<kronicle provided password>
```

And the contents of your details conf file looks like:
```
[publishing_details]
visibility=<private or public>
embargo_months=<up to 24, ignored if visibility=public>
publishing_info=<optional path to a bibtex.tex entry for publication reference>
publisher_name=<name of the individual running rims publish>
publisher_email=<email of the individual running rims publish>
publisher_orcid=<orcid ID of the individual running rims publish>
maintainer_name=<name of the individual/team responsible for generating the rims products being uploaded>
maintainer_email=<email of the individual/team responsible for generating the rims products being uploaded>
maintainer_orcid=<orcid ID of the individual/team responsible for generating the rims products being uploaded>
```

For details on locally creating an upload server to host your dynamic spectra products, visit [rims-upload-server](https://github.com/mhardcastle/rims-upload-server). Otherwise, contact authors for a server to which you may upload your dynamic spectra.

### Licensing:
MIT License

Copyright (c) 2017-2025, Cyril Tasse, Alan Loh, Martin Hardcastle, Talon Myburgh, Observatoire de Paris, University of Hertfordshire, Rhodes University
