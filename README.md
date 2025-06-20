# Extraction of Elemental Maneuvers using LDA

## Environment

   Environment of the developer
   - OS
     - ProductName: macOS
     - ProductVersion: 15.4.1
     - BuildVersion: 24E263
   - CPU
     - Apple M3
   - X code
     - Xcode 16.4
     - Build version 16F6
   - conda virtual env (picked up)
     - cython==3.0.12
     - gensim==4.3.0
     - meson-python==0.18.0
     - ninja==1.12.1
     - numpy==1.21.2
     - openpyxl==3.0.9
     - pandas==1.3.3
     - python==3.9.9
     - scikit-learn==1.5.2 (installed by the method below)
     - scipy==1.10.1
     - seaborn==0.13.2
     - setuptools==59.8.0
     - tqdm==4.67.1

## Usage

### Preparation
   1. Move raw data to `/raw_data/`
   2. Build sklearn
   
      See https://scikit-learn.org/stable/developers/advanced_installation.html    

   3. Move to `/src/`
   4. Compile fortran
      
      Run `bash ./setup.sh`

### Main

   Run in the following order

   1. `data.py`
   2. `quantizetion.py`
      - K_EACH
        - number interpreted as the division number in each dimension of self.TARGET_ELEMENT
        - number of vocabulary in codebook is defided as $\mathrm{K\_EACH}^{\mathrm{len(\mathrm{TARGET\_ELEMENT})}}$
   3. `segmentation.py`
      - L_DOC
        - length of a segment
      - DELTA_TS_SHIFT
        - number of time steps shifted to the start of the next document
   4. `LDA_sklearn.py`
      - N_TOPIC
        - number of topics: $K$


## Data

### Maneuvering Time Series Data
provided by Dr. Ishibashi

### kml file of Ise Bay

downloaded from [Google Earth](https://earth.google.com/web/@34.82410736,136.89131379,39.06159975a,238429.02075412d,30y,0h,0t,0r/data=CgRCAggBMikKJwolCiExTzAwREFxcDFGdGRqb3VINWhvUkNoYURjYkN2N25jdEggAToDCgEwQgIIAEoICMi3ysMHEAE)

