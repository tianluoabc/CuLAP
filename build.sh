# Complies the files and creates the Linker
# Todo - Test on several different architectures - 
mkdir ./lib
cd ./include
nvcc -c *.cu
ar rcs culap.a *.o
mv ./culap.a ../lib/libculap.a
rm -rf *.o # cleanup
cd ..
