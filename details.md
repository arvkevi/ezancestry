# Details

Depending on which company performed your genotyping, your sample might be missing information for some of the 55 (or 128) **AISNPs**.

* To show a table of your genotypes at the AISNP locations, check **Show Your Genotypes**.

* To show a table of the 1000 genomes samples genotypes at the AISNP locations, check **Show 1k Genomes Genotypes**.

## 1000 Genomes and AISNPs Data

A subset of 1000 Genomes Project samples' single nucleotide polymorphism(s), or, SNP(s) have been parsed from the [publicly available `.bcf` files](ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/bcf_files/).  
The subset of `SNPs` were chosen from two publications which identified **AISNPs**:
  * Set of 55 AISNPs. [Progress toward an efficient panel of SNPs for ancestry inference](https://www.ncbi.nlm.nih.gov/pubmed?db=pubmed&cmd=Retrieve&dopt=citation&list_uids=24508742). Kidd et al. 2014
  * Set of 128 AISNPs. [Ancestry informative marker sets for determining continental origin and admixture proportions in common populations in America.](https://www.ncbi.nlm.nih.gov/pubmed?cmd=Retrieve&dopt=citation&list_uids=18683858). Kosoy et al. 2009 (Seldin Lab)

## Parameters

### Set of AISNPs to use

* `Kidd 55 AISNPs`: Subset the 1kG data to the 55 SNPs listed in the manuscript.
* `Seldin 128 AISNPs`: Subset the 1kG data to the 128 SNPs listed in the manuscript.

### Population Resolution

* `Super Population`: One of {AFR, AMR, EAS, EUR, SAS}.
* `Population`: One of the 26 populations listed [here](http://www.internationalgenome.org/faq/which-populations-are-part-your-study/).

## Code

The code used to process this data is available on [GitHub](https://github.com/arvkevi/ezancestry).
