# nnsvs-db-converter
 Python script to convert NNSVS DBs to Diffsinger without needing the NNSVS Python Library.
# Information
 This is a Python script for converting singing databases with HTS mono labels into the Diffsinger format. This Python script deals with sample segmentation to ensure that samples have a maximum given length and a maximum amount of pauses in between. The only recognized silence phoneme by this script is `pau` and also assumes the labels have silences labeled at the start and end.
 
 This Python script only requires numpy and soundfile to work, unlike the existing one which needs the NNSVS library, which might be hard to install for most people.
# How to Use
 1. Have a Python install, preferably Python 3.8.
 2. Install the `numpy` and `soundfile` libraries through pip like so:
 ```pip install numpy soundfile```
 3. Pass the database with either method:
    1. Drag and drop the database folder over the script.
    2. Run the script using the terminal and pass terminal arguments.
    
```
usage: db_converter.py [-h] [--max-length float] [--max-silences int] [--write-labels] [--debug] path

Converts a database with mono labels (NNSVS Format) into the DiffSinger format and saves it in a new folder in the
path supplemented.

positional arguments:
  path                  The path of the folder of the database.

optional arguments:
  -h, --help            show this help message and exit
  --max-length float, -l float
                        The maximum length of the samples in seconds. (default: 15)
  --max-silences int, -s int
                        The maximum amount of silences (pau) in the middle of each segment. Set to a big amount to
                        maximize segment lengths. (default: 0)
  --write-labels, -w    Write Audacity labels if you want to check segmentation labels. (default: False)
  --debug, -d           Show debug logs. (default: False)
```