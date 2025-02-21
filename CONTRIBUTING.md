Contributing
------------

Please use the following guidelines when contributing to this project. 

Before contributing signficicant changes, please begin a discussion of the
desired changes via a GitHub Issue to prevent doing unnecessary or overlapping
work.

## License

The preferred license for contributions to this project is the detailed in the 
[LICENSE file](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/LICENSE).  

Contributions must include a "signed off by" tag in the commit message for the
contributions asserting the signing of the developers certificate of origin
(https://developercertificate.org/). A GPG-signed commit with the "signed off
by" tag is preferred.

## Styling

Please use the guidelines detailed in the [notebook-template.ipynb](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/notebook-template.ipynb) for contributions.

## Contributing Labs/Modules

A module should have the following directory structure:

* The base of the module should contain a README.ipynb file with a brief
  introduction to the module and links to the individual labs for each
  language translation and programming language available.
* The base of the module should contain a subdirectory for each programming language if applicable. Each
  of these directories should contain a directory for each language
  translation provided (English, for instance).
* The base of the module should contain an `images` directory that contains
  images that will be used in common between multiple notebooks. 
* For each programming language and translation there should be a file named
  `README.ipynb` containing the actual lab instructions. A single file name
  is used in this way to simplify finding the starting point for each new
  lab.
* Each lab translation and programming language combination should have a
  `solutions` directory containing correct solutions.

## Attribution

Portions adopted from [https://github.com/OpenACC/openacc-training-materials/blob/master/CONTRIBUTING.md](https://github.com/OpenACC/openacc-training-materials/blob/master/CONTRIBUTING.md)