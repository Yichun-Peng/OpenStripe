---

OpenStripe
OpenStripe: Autonomous Open-set Discovery of Amur Tiger Individuals via Multi-agent Stripe Reconstruction

This is the official open-source repository for the paper: "Autonomous open-set discovery of Amur tiger individuals via multi-agent stripe reconstruction".

Project Overview
This project provides a technical framework for the individual identification and "open-set discovery" of wild Amur tigers (*Panthera tigris altaica*). Utilizing a **multi-agent blackboard architecture**, the system automates stripe reconstruction, feature extraction, and individual clustering within complex wild environments.

Data Open-Source Statement (Important)
The original images and videos for this study were collected from **sensitive border regions between China, North Korea, and Russia. To comply with security protocols and protect border geographic information and patrol dynamics, the data has undergone the following desensitization process. Only code and non-sensitive data are made public here:
Raw Data Concealment: Original infrared camera videos and images containing environmental backgrounds are not publicized.
Position Desensitization: Precise GPS coordinates and specific camera trap deployment locations are withheld.
Publicly Available Content: We provide only the **binary stripe skeleton maps (located in `reconstructed_patterns/`) processed by the multi-agent system, along with the extracted anonymized feature vectors** (`.npz` files).

File Structure
`openstripe_mas_demo.py`: Example code for the multi-agent collaborative framework (API keys and sensitive prompts have been removed).
`plot_fig2ab.ipynb`: Notebook for reproducing the individual feature distance matrix and Kernel Density Estimation (KDE) distribution plots.
`plot_fig2c.ipynb`: Notebook for reproducing the circular dendrogram of the clustering results.
`plot_fig3b_analysis.py`: Script for the correlation analysis between geographic distance and phenotypic similarity.

Environment Requirements
Python 3.10+
numpy, scipy, matplotlib, seaborn, scikit-learn

Citation
If you use this project in your research, please cite our paper:
> Yincalos Peng, et al. "Autonomous open-set discovery of Amur tiger individuals via multi-agent stripe reconstruction." 

---
