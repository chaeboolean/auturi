#!bin/bash

# gfootball
GRF_VER=v2.8
GRF_PATH=football/third_party/gfootball_engine/lib
GRF_URL=https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_${GRF_VER}.so
git clone -b ${GRF_VER} https://github.com/google-research/football.git
mkdir -p ${GRF_PATH}
wget -q ${GRF_URL} -O ${GRF_PATH}/prebuilt_gameplayfootball.so
cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install . && cd ..
