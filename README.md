 # 🦷 Dental Keypoint Pipeline
 
 This repository contains the steps and code required to perform the **dental keypoint pipeline**, as presented in the papers:  
- _Kofod Petersen, A., Forgie, A., Bindslev, D. A., Villesen, P. & Staun Larsen, L. **Automatic removal of soft tissue from 3D dental photo scans; an important step in automating future forensic odontology identification**. Scientific Reports 14, 12421 (2024). https://doi.org/10.1038/s41598-024-63198-2_.
- _Kofod Petersen, A., Forgie, A., Villesen, P. & Staun Larsen, L. **3D Dental Similarity Quantification in Forensic Odontology Identification**. Forensic Science International, 112462 (2025). https://doi.org/10.1016/j.forsciint.2025.112462_.
- _Kofod Petersen, A. et al. **Charred or Fragmented, Yet Comparable: Quantifying Dental Surface Similarity Across Teeth, Jaws, and Heat Exposure**. medRxiv (preprint), 2025.2004.2007.25325362 (2025). https://doi.org/10.1101/2025.04.07.25325362_.
- _Kofod Petersen, A., Arenholt Bindslev, D., Forgie, A., Villesen, P. & Staun Larsen, L. **Objective comparison of 3D dental scans in forensic odontology identification**. medRxiv (preprint), 2025.2003.2031.25324929 (2025). https://doi.org/10.1101/2025.03.31.25324929_.
- _Kofod Petersen, A., Spin-Neto, R., Villesen, P. & Staun Larsen, L. **Curvature-based 3D dental comparison to identify trauma-induced surface changes in human teeth: A forensic comparison study**. medRxiv (preprint), 2025.2006.2019.25329914 (2025). https://doi.org/10.1101/2025.06.19.25329914_.


 
 
 ---
 
 ## 🚀 Setup Environment  
 
 It is recommended to use a **Conda environment** for this project.  
 
 ### 1️⃣ Clone the Repository  
 ```sh
 git clone https://github.com/AnikaKofodPetersen/Dental_keypoint_pipeline.git
 cd Dental_keypoint_pipeline
 ```
 
 ### 2️⃣ Create and Activate the Conda Environment  
 ```sh
 conda env create -f environment.yml
 conda activate AKPpyshot_env
 ```
 
 ---
 
 ## 🛠️ Install the PySHOT Library  
 
 To install the **PySHOT** library, follow these steps (originally described [here](https://github.com/uhlmanngroup/pyshot)):  
 
 ### 1️⃣ Clone the PySHOT Repository  
 ```sh
 git clone https://github.com/uhlmanngroup/pyshot.git
 ```
 
 ### 2️⃣ Ensure Index Integrity for SHOT Descriptors  
 Copy the `shot_descriptor.cpp` file before building:  
 ```sh
 cp ./shot_descriptor.cpp pyshot/src/shot_descriptor.cpp
 ```
 
 ### 3️⃣ Build and Install PySHOT  
 ```sh
 cd pyshot
 pip install .
 ```
 
 ---
 
 ## 📓 Jupyter Notebook Support  
 
 If you plan to work with **Jupyter notebooks**, install the necessary dependencies and register the environment as a kernel:  
 ```sh
 conda install pip
 conda install ipykernel
 python -m ipykernel install --user --name AKPpyshot_kernel
 ```
 
 ---
 
 ## 📖 Getting Started  
 
 To begin:  
 1. Read the papers referenced at the beginning of this README.
 2. Install jupyter in your conda environment
 ```sh
cd ..
conda install jupyter
pip install --force-reinstall charset_normalizer
 ```
 3. Run jupyter notebook
 ```sh
 jupyter notbeook
  ```
 4. Follow the **tutorial notebook** included in this repository.  
    - It was automatically downloaded when you cloned this repo.  
 
Happy coding! 🎯  

