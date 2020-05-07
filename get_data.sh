#!/bin/bash

data_files = [https://upenn.box.com/shared/static/jndes1jbjogj045ulq0q6g4hp3tmepdl,https://upenn.box.com/s/0xuew7f5uplqopy9hu95w2tw6me8ct3e]

mkdir data
cd data

mkdir train
cd train
curl -L https://upenn.box.com/shared/static/c3bszne8algiclmblixl51lp37moojzq --output 00.zip

curl -L https://upenn.box.com/shared/static/jndes1jbjogj045ulq0q6g4hp3tmepdl --output 01.zip

curl -L https://upenn.box.com/shared/static/0xuew7f5uplqopy9hu95w2tw6me8ct3e --output 02.zip

curl -L https://upenn.box.com/shared/static/m2rhqdujoeovk5x5v8fn9g0e1r4txxb7 --output 03.zip

curl -L https://upenn.box.com/shared/static/3kxkrwc2k57dofy52xkxcostc95lt90a --output 04.zip 

curl -L https://upenn.box.com/shared/static/fcx86lq1k2v3nxf130jaeyfubuxz52vz --output 05.zip 

curl -L https://upenn.box.com/shared/static/a7o4tyqq2mgppqcift0325zqwazy19i1 --output 06.zip 

curl -L https://upenn.box.com/shared/static/a5nnbc0y8qx4syk6n0o589i0sc4mqjek --output 07.zip

unzip -q 00.zip
unzip -q 01.zip
unzip -q 02.zip
unzip -q 03.zip
unzip -q 04.zip
unzip -q 05.zip
unzip -q 06.zip
unzip -q 07.zip

rm 01.zip
rm 02.zip
rm 03.zip
rm 04.zip
rm 05.zip
rm 06.zip
rm 07.zip

cd ../
mkdir val
cd val
curl -L https://upenn.box.com/shared/static/boh0gckwid1jrcl7gyl5ne3qdv621fef --output 08.zip

curl -L https://upenn.box.com/shared/static/kem45j2tzyb21trkp25asshf18yiu5hs --output 09.zip

curl -L https://upenn.box.com/shared/static/k6iymlpfms0aa2qrd3j5oh6l7ucfuzq7 --output 10.zip

unzip -q 08.zip
unzip -q 09.zip
unzip -q 10.zip

rm 08.zip
rm 09.zip
rm 10.zip

cd ../
curl -L https://upenn.box.com/shared/static/x7yfgwqd9pn02di6eakcnubafxplp08y --output final.zip

unzip -q final.zip
rm final.zip

pip3 install efficientnet_pytorch
