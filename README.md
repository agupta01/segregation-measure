# segregation-measure

This is the code for the project "Quantifying the Effects of Redlining on Racial Segregation in America's Urban Centers". Below you will find information on how to build the data product and reproduce the findings of the study.

## Requirements

- Anaconda
- Python (>= 3.6)
- Lots of RAM (if you're rebuilding all data sources, > 16GB is preferred)

See `environment.yml` for more details on the package requirements.


## Getting Started

1. Clone this repository.
2. Build the anaconda environment using the following command: `conda env create -f environment.yml`.
3. Download and unzip the data package from [here](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/arg002_ucsd_edu/EVdq6ZXbCM5Jv8r6cjs__b4BmSMWQ_WjrsxvmY1ChrfSMg?e=3LyX7j). (Or, if you're feeling particularly adventurous, build the data product yourself by getting the data from IPUMS and using `redlining-maps-crosswalks-2010.ipynb`)
4. Pick a city and run the chains on it with the following command: `python measure-parallel.py <CITY> <STATE> <FIPS>`.
```
usage: measure-parallel.py [-h] [-s STEPS]
                           [-w WORKERS]
                           city state fips

positional arguments:
  city                  city name, i.e. Atlanta
  state                 state code, i.e. GA
  fips                  state FIPS code (zero-
                        padded on the end), i.e.
                        130

optional arguments:
  -h, --help            show this help message
                        and exit
  -s STEPS, --steps STEPS
                        number of steps for each
                        markov chain
  -w WORKERS, --workers WORKERS
                        total # of worker
                        processes across both
                        proposals
```
NOTE: If you wish to simply try out the code, use the flag `-w 2` to minimize the number of threads you'll have to manually kill if things go wrong.
If you are running to reproduce, use `-s 1000000` and make sure the random seed is set to `2020` (line 306 in `measure-parallel.py`). Note that this will take a VERY long amount of time, and thus it's only recommended to be run on a cloud instance or a very powerful machine.

## Acknowledgements
Author: Arunav Gupta (arunavg (at) ucsd (dot) edu)

Advisor: Dr. Isaac Martin, Urban Studies and Planning Department, UC San Diego

Project was funded in part by the 2020 Halicioglu Data Science Undergraduate Research Scholarship.

This project was heavily inspired by the work of Dr. Moon Duchin and [Metric Geography and Gerrymandering Group](https://mggg.org/), as well as the [Racial Dot Map](http://racialdotmap.demographics.coopercenter.org/) from the University of Virginia.

Data Sources:

IPUMS NHGIS: Steven Manson, Jonathan Schroeder, David Van Riper, and Steven Ruggles. *IPUMS National Historical Geographic Information System*: Version 14.0 [Database]. Minneapolis, MN: IPUMS. 2019. [http://doi.org/10.18128/D050.V14.0](http://doi.org/10.18128/D050.V14.0).

*Mapping Inequality*: Robert K. Nelson, LaDale Winling, Richard Marciano, Nathan Connolly, et al., “Mapping Inequality,” *American Panorama*, ed. Robert K. Nelson and Edward L. Ayers, accessed February 23, 2021, [https://dsl.richmond.edu/panorama/redlining/](https://dsl.richmond.edu/panorama/redlining/).

