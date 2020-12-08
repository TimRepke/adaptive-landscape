# Compile externals
cd ../Code/externals/bhtsne || exit
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
chmod +x bh_tsne

cd ../FItSNE || exit
g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  -o bin/fast_tsne -pthread -lfftw3 -lm -Wno-address-of-packed-member

cd ../LargeVis/Linux || exit
g++ LargeVis.cpp main.cpp -o LargeVis -lm -pthread -lgsl -lgslcblas -Ofast -march=native -ffast-math
pip install --editable .

cd ../../openTSNE || exit
python setup.py build_ext --inplace
pip install --editable .