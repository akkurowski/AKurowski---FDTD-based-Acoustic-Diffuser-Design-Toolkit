mkdir Data
mkdir Data/GPU0
mkdir Data/GPU1

mv    GPU0/_stored_data/* Data/GPU0/
rm    Data/GPU0/diffuser.npy

mv    GPU1/_stored_data/* Data/GPU1/
rm    Data/GPU1/diffuser.npy
 
compress-archive Data diff_impres_data.zip
