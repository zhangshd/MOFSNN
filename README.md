# MOF Stability prediction Neural Network (MOFSNN)
This is a study of predicting various types of MOF stability using multi-task strategy.
## Raw data
The raw data is stored in the `raw_data` folder, which can get from literature. 
### TSD & SSD 
The thermal stability dataset and solvent stability dataset are get from [Nandy's work](https://pubs.acs.org/doi/10.1021/jacs.1c07217).
The data is download from [here](https://zenodo.org/records/5737968/files/SciData.zip?download=1), which is published in [Nandy's another paper](https://www.nature.com/articles/s41597-022-01181-0)
The CIF files are matched with MOFs in the CoREMOF2019 databse, which is available from [MOFX-DB](https://mof.tech.northwestern.edu/databases).
### WS24
The water stability, acid stability, base stability, and boiling stability dataset are get from [Terrones's work](https://pubs.acs.org/doi/10.1021/jacs.4c05879).
And the raw data is download from [here](https://zenodo.org/records/12110918).
### Download and extract data
```sh
wget https://zenodo.org/records/5737968/files/SciData.zip?download=1 -O Nandy_2022_SciData.zip
wget https://zenodo.org/api/records/12110918/files-archive -O WS24v2.zip
wget https://mof.tech.northwestern.edu/Datasets/CoREMOF%202019-mofdb-version:dc8a0295db.zip -O CoREMOF2019.zip
unzip Nandy_2022_SciData.zip
unzip WS24v2.zip
unzip CoREMOF2019.zip
```
