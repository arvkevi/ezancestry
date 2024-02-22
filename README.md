# ezancestry

![Build](https://github.com/arvkevi/ezancestry/actions/workflows/ci.yml/badge.svg)  

Easily visualize your direct-to-consumer genetics next to 2500+ samples from the 1000 genomes project. Evaluate the performance of a custom set of ancestry-informative snps (AISNPs) at classifying the genetic ancestry of the 1000 genomes samples using a machine learning model.

A subset of 1000 Genomes Project samples' single nucleotide polymorphism(s), or, SNP(s) have been parsed from the [publicly available `.bcf` files](ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/bcf_files/).  
The subset of `SNPs`, AISNPs (ancestry-informative snps), were chosen from two publications:

* Set of 55 AISNPs. [Progress toward an efficient panel of SNPs for ancestry inference](https://www.ncbi.nlm.nih.gov/pubmed?db=pubmed&cmd=Retrieve&dopt=citation&list_uids=24508742). Kidd et al. 2014
* Set of 128 AISNPs. [Ancestry informative marker sets for determining continental origin and admixture proportions in common populations in America.](https://www.ncbi.nlm.nih.gov/pubmed?cmd=Retrieve&dopt=citation&list_uids=18683858). Kosoy et al. 2009 (Seldin Lab)

ezancestry ships with pretrained k-nearest neighbor models for all combinations of following:

    * Kidd (55 AISNPs)
    * Seldin (128 AISNPs)
    
    * continental-level population (superpopulation)
    * regional population (population)
 
    * principal component analysis (PCA)

![image](images/ezancestry.gif)

## Table of Contents

* [Installation](#installation)
* [Config](#config)
* [Usage](#usage)
  * [command line tool](#command-line-interface)
    * [fetch](#fetch)
    * [predict](#predict)
    * [plot](#plot)
    * [train](#train)
  * [Python API](#python-api)
* [Visualization](#visualization)
* [Contributing](#contributing)

## Installation

Install ezancestry with pip:

```shell
pip install ezancestry
```

Or clone the repository and run `pip install` from the directory:

```shell
git clone git@github.com:arvkevi/ezancestry.git
cd ezancestry
pip install .
```

## Config

The first time `ezancestry` is run it will generate a `config.ini` file and `data/` directory in your home directory under `${HOME}/.ezancestry`.
You can edit `conf.ini` to change the default settings, but it is not necessary to use ezancestry. The settings are just a utility for the user so they don't have to be verbose when interacting with the software. The settings are also keyword arguments to each of the commands in the ezancestry API, so you can always override the default settings.  

These will be created in your home directory:

```shell
${HOME}/.ezancestry/conf.ini
${HOME}/.ezancestry/data/
```

Explanations of each setting is described in the Options section of the `--help` of each command, for example:

```shell
ezancestry predict --help

Usage: ezancestry predict [OPTIONS] INPUT_DATA

  Predict ancestry from genetic data.

  * Default arguments are from the ~/.ezancestry/conf.ini file. *

Arguments:
  INPUT_DATA  Can be a file path to raw genetic data (23andMe, ancestry.com,
              .vcf) file, a path to a directory containing several raw genetic
              files, or a (tab or comma) delimited file with sample ids as
              rows and snps as columns.  [required]


Options:
  --output-directory TEXT         The directory where to write the prediction
                                  results file

  --write-predictions / --no-write-predictions
                                  If True, write the predictions to a file. If
                                  False, return the predictions as a
                                  dataframe.  [default: True]

  --models-directory TEXT         The path to the directory where the model
                                  files are located.

  --aisnps-directory TEXT         The path to the directory where the AISNPs
                                  files are located.

  --aisnps-set TEXT               The name of the AISNP set to use. To start,
                                  choose either 'Kidd' or 'Seldin'. The
                                  default value in conf.ini is 'Kidd'. *If
                                  using your AISNP set, this value will be the
                                  in the namingc onvention for all the new
                                  model files that are created*

  --help                          Show this message and exit.
```

## Usage

ezancestry can be used as a command-line tool or as a Python library.

`ezancestry predict` comes with pre-trained models when `--aisnps-set=kidd` or `--aisnps-set=seldin`.`

![image](images/ezancestry.drawio.png)

### command-line interface

There are four commands available:

1. `fetch`: generate a csv file with all the 1000 Genome samples (rows) at the specified AISNPs locations (columns).
2. `predict`: predict the genetic ancestry of a sample or cohort of samples using the nearest neighbors model.
3. `plot`: plot the genetic ancestry of samples using the output of `predict`.
4. `train`: build a k-nearest neighbors model from the 1000 genomes data using a custom set of AISNPs.

Use the commands in the following way:

#### predict

ezancestry can predict the genetic ancestry of a sample or cohort of samples using the nearest neighbors model.
The `input_data` can be a file path to raw genetic data (23andMe, ancestry.com, .vcf) file, a path to a directory containing several raw genetic files, or a (tab or comma) delimited file with sample ids as rows and snps as columns.

This writes a file, `predictions.csv` to the `output_directory` (defaults to current directory). This file contains predicted ancestry for each sample.

**Direct-to-consumer genetic data file (23andMe, ancestry.com, etc.)**:

```shell
ezancestry predict mygenome.txt
```

**Directory of direct-to-consumer genetic data files or .vcf files**:

```shell
ezancestry predict /path/to/genetic_datafiles
```

**comma-separated file with sample ids as rows and snps as columns, filled with genotypes as values**

```shell
ezancestry predict ${HOME}/.ezancestry/data/aisnps/thousand_genomes.KIDD.dataframe.csv
```

#### plot

Visualize the output of `predict` using the `plot` command. This will open a 3d scatter plot in a browser.

```shell
ezancestry plot predictions.csv
```

#### fetch

This command will download all of the data required to build a new nearest neighbors model for a custom set of AISNPs. If you want to use existing models, see `predict` and `plot`.

Without any arguments this command will download all necessary data to build new models and put it in the `${HOME}/.ezancestry/data/` directory.

```shell
ezancestry fetch
```

Now you are ready to build a new model with `train`.

#### train

Test the discriminative power of your custom set of AISNPs.

This command will build all the necessary models to visualize and predict the 1000 genomes samples as well as user-uploaded samples. A model performace evaluation report will be generated for a five-fold cross-validation on the training set of the 1000 genomes samples as well as a report for the holdout set.

Create a custom AISNP file here: `~/.ezancestry/data/aisnps/custom.AISNP.txt`. The prefix of the filename, `custom`, can be whatever you want. Note that this value is used as the `aisnps-set` keyword argument for other ezancestry commands.

The file should look like this:

```
id      chromosome      position
rs731257        7       12669251
rs2946788       11      24010530
rs3793451       9       71659280
rs10236187      7       139447377
rs1569175       2       201021954
```

```shell
ezancestry train --aisnps-set=custom
```

### Python API

See the [notebook](github.com/arvkevi/ezancestry/ezancestry_library_tutorial.ipynb)

### Visualization

[Open in Streamlit](https://share.streamlit.io/arvkevi/ezancestry/streamlit/app.py)

![image](images/ezancestry.png)

### Contributing

Contributions are welcome! Please feel free to create an issue for discussion or make a pull request.
