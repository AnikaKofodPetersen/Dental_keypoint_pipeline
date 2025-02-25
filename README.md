 # 🦷 Dental Keypoint Pipeline
 
 This repository contains the steps and code required to perform the **dental keypoint pipeline**, as presented in the papers: _[Add references here]_.  
 
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
 2. Follow the **tutorial notebook** included in this repository.  
    - It was automatically downloaded when you cloned this repo.  
 
Happy coding! 🎯  

