# StrainProx
This repository contains reprodible material for: StrainProx: High resolution seismic time-strain estimation


## Project Structure
The repository is organized as follows:

* :open_file_folder: **strainprox**: A Python library that includes routines for the invesion algorithms used and plotting functions.
* :open_file_folder: **data**: A folder containing the data or instructions on how to obtain it.
* :open_file_folder: **notebooks**: A collection of Jupyter notebooks organized in several folders that document the use of StrainProx for time-strain inversion with Hess model.



## Getting Started :space_invader: :robot:
To reproduce the results, use the `environment.yml` file for environment setup.

Execute the following command:
```
./install_env.sh
```
The installation takes some time. If you see `Done!` in your terminal, the setup is complete.

Finally, run:
```
pip install -e . 
```
in the folder where the setup.py file is located.


Always activate the environment with:
```
conda activate strainprox
```



