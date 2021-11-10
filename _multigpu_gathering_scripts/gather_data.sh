mkdir DataGPU0
mkdir DataGPU1

mv    GPU0/_stored_data/* DataGPU0
rm    DataGPU0/diffuser.npy

mv    GPU1/_stored_data/* DataGPU1
rm    DataGPU1/diffuser.npy
 
zip   data.zip ./DataGPU0/* ./DataGPU1/*