# new_compute is used to store the generated TVM compute code.
#to use p2T, run the following command, where xx has to be replaced with the kernel name and the optimization option opt takes the value of padding, shifting or none

python3 P2T.py xx_polyhedral_model.txt xx_array_size.txt opt

#For instance, for the polynomialmul kernel and the optimization padding, the following command generated the TVM compute

python3 P2T.py polynomialmul_polyhedral_model.txt polynomialmul_array_size.txt padding


