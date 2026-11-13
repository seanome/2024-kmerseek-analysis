# Inputs for the exercise score (notebook 361, written 2026-11-03)

Domen et al. 2026 Fig. 4C-D shows a score in 8 exercise datasets. The panels are "adopted from"
their ref 43, Voskoboynik, McCulloch and Sahoo 2024, Mol Metab 90:102058
(https://doi.org/10.1016/j.molmet.2024.102058). The genes come from Ghosh et al. 2023, eBioMedicine
(https://doi.org/10.1016/j.ebiom.2023.104719).

- `cluster_13.genes.txt`, `cluster_14.genes.txt`, `cluster_3.genes.txt`: the 338 genes, 48 + 137 +
  153. Cluster 13 is the group called pro-immune (M1); clusters 14 and 3 are the group called
  cell-repair (M2). Copied from https://github.com/YoyoVosko/MacrophagesOntheRun, commit 4cf2f16f,
  `Files/Mac_Cluster_*_Genes.txt`. They are the same genes as `SMaRT/node-*.txt` in
  https://github.com/sahoo00/BoNE, commit b257268e.
- `voskoboynik2024_table_s1.csv`: AUC and p values from the same repository,
  `Files/SupplementaryTable1_Immediate_and_Training_Datasets_ROCAUC_Pvalues.csv`.
- `published_fig4.tsv`: numbers typed from the Fig. 4C-D panels of Domen et al. 2026. The figure
  was read from a screenshot Olga pasted into the chat on 2026-11-03. Panel GSE83578 prints no p for
  24 h. GSE27536 uses cluster 13 alone, as the paper's methods say. The two last rows, panel
  "Table S1", come from `voskoboynik2024_table_s1.csv`.
- `expected_probe_counts.tsv`: probes per cluster that the authors' notebook printed for each dataset
  (`Macrophages_On_The_Run_Make_Figures_Code.ipynb`, Figure 3 cell). Blank means that dataset's cell
  ran cluster 13 alone.
