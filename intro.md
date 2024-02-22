# Genetic Ancestry Visualization

The figure below displays samples from the [1000 Genomes Project](http://www.internationalgenome.org/home). What you are seeing are the samples' genotypes projected into a three-dimensional feature-space through dimensionality reduction techniques. Data points are colored according to a sample's [reported genetic ancestry](http://www.internationalgenome.org/faq/which-populations-are-part-your-study/).

The genotypes were filtered to include only a small subset of the genome called [ancestry-informative single nucleotide polymorphisms](https://en.wikipedia.org/wiki/Ancestry-informative_marker) (**AISNPs**). Then, the genotypes were one-hot encoded. Finally, dimensionality reduction was performed to facilitate visualization.

**You can also visualize your direct-to-consumer genetic results (e.g. [23andMe](https://customercare.23andme.com/hc/en-us/articles/212196868-Accessing-Your-Raw-Genetic-Data)). A k-nearest neighbors classifier will predict which of the populations you are most closely related to based on the AISNP genotypes.**
