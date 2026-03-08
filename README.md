# DSC180B-B05-1-Capstone

Steps for replication:

1. In the parent folder, clone the following repository: https://github.com/Nolancchu/DSC180B-B05-1-Capstone

2. Download and place the folder linked into the cloned repo: https://www.dropbox.com/scl/fo/ywr2qznkmlr4jrkl859p8/AJZA_MsE3GkQnjGF5cnUAmA?rlkey=57a8x14jzwfi8wjlhu2n6mqcb&e=1&st=1xbk8yz4&dl=0

3. Ensure that you have all the libraries specified in requirements.txt. Use "pip install -r requirements.txt" in order to download them all at once. 

4. In the DSC-180B-B05-1-Capstone folder, run all cells in sea_level_testing.ipynb in order to generate projections csv files.

5. In the BRICK folder, run generate_rslr.r in order to generate the regional sea level rise projections for each pathway.

6. For the final model, download sliiders-v1.2.zar.zip at link: https://zenodo.org/records/10909655, open the projected costs folder and run file: run_pyciam_regional_rise.ipynb.

6. To generate the visualizations, first download https://drive.google.com/file/d/1Dk3DYOhaYahwIBE-Kxa8t4NTuqE8d1fL/view?usp=sharing and place it in the repository. Do the same with this folder https://drive.google.com/file/d/1VW4dHIp7_befuztdh3rXQqmqt-TfU901/view?usp=sharing. This contains the packages required for the map visualizations of costs. Lastly, run visualizations_merged.ipynb.
