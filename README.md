# tcell_migration

This code was developed for the analysis of data in:

<h3>T cells Use Focal Adhesions to Pull Themselves Through Confined Environments</h3><br>
Caillier&nbsp;A, Oleksyn&nbsp;D, Fowell&nbsp;DJ, Miller&nbsp;J, Oakes&nbsp;PW<sup>&#9993;</sup><br>
<a class="unadorned" target="_blank" href="https://doi.org/10.1101/2023.10.16.562587">DOI</a>  | <a class="unadorned" target="_blank" href="https://www.biorxiv.org/content/10.1101/2023.10.16.562587v1">bioR&chi;iv</a> | <a class="unadorned" target="_blank" href="https://x.com/pwoakes/status/1715169434061717897?s=20"><img src="images/Twitter_Social_Icon_Circle_Color.png" class="social"> Twitter thread</a> | <a class="unadorned" target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/37904911/">PMID: 37904911</a><br>

All code is written in python and used the following packages (version tested on included in parantheses):
- python (3.9.18)
- jupyter (1.0.0)
- numpy (1.26.3)
- skimage (0.22.0)
- trackpy (0.6.1)
- matplotlib (3.8.2)
- seaborn (0.13.2)
- scipy (1.12.0)

The code also calls ffmpeg from the command line to make movies - though this is strictly optional

# Example Notebooks

### Tcell_tracking_example.ipynb

This notebook gives a basic walkthrough of tracking analysis using 10X images of T cells stained with Hoechst (to label nuclei). Trackpy is used as the basis for identifying and linking particle trajectories

### Tcell_follow_the_leader_analysis_example.ipynb

This notebook uses the output from above to analyze cell trajectories for potential overlapping regions. 