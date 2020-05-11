# Genetic Ancestry Visualization
The figure below displays samples from the [1000 Genomes Project](http://www.internationalgenome.org/home). The samples' genotypes are projected into a three-dimensional feature-space through dimensionality reduction techniques. Data points are colored according to a sample's [reported genetic ancestry](http://www.internationalgenome.org/faq/which-populations-are-part-your-study/).

The genotypes filtered to include only ancestry-informative single nucleotide polymorphisms(**AISNPs**). Then they were one-hot encoded, and finally dimensionality reduction was performed. Projecting the genotypes into three dimensions makes it easy to visualize the structure of the 1000 Genomes Project samples in the context of **AISNPs**.

**You can also visualize your direct-to-consumer genetic results (e.g. 23andMe). A k-nearest neighbors classifier will predict which of the populations you are most closely related based on the AISNP genotypes.**