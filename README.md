#### ASD-DiagNet: A hybrid learning approach for detection of Autism Spectrum Disorder using fMRI data
https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2019.00070/full

##### Dependencies
Python 3.9 
Numpy ==
Nilearn == 
Tensorflow ==
Scikit-Learn ==

##### Models
- We have two models both work on detection of Autism Spectrum Disorder (ASD) 
- First Model works on: 
  - Pipeline: C-PAC 
  - Data Type: Preprocessed fMRI time-series (func_preproc)
  - Features: Functional connectivity matrices (correlation matrices)
  - Labels: Diagnosis labels (DX_GROUP), where:
    - 1 indicates ASD (Autism Spectrum Disorder)
    - 2 indicates Control (Healthy)
  - Number of Subjects: 100 (by default, but you can change this)
  - Input: nii or nii.gz
  - If you have image only convert DICOM files to NIfTI format
- 
