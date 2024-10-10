#According to the official website https://pluto-compiler.sourceforge.net/ After installing the Plut compiler, run the replace.sh script to enhance the functionality of Pluto
# pluto version 
pluto-0.11-4

# please ensure src_replace and pluto-0.11.4 are in a unified file directory
mv src_replace/ pluto-0.11.4/
cd pluto-0.11.4/
cd src_replace/
run bash replace.sh
cd ..
# recompile Pltuo
make -j4 
# create your test
mkdir your_test/
cd your_test/

# Put your kernels into the your_test folder.
# run pluto command to get polyhredral model
../polycc your_kernel.c

#this will generate a xx_polyhedral_model.txt file in your_test,please move it to P2T file so that you can convert it to compute
