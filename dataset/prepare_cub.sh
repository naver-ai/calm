#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45" -O dataset/CUB_200_2011.tgz && rm /tmp/cookies.txt
tar xvf dataset/CUB_200_2011.tgz -C dataset/
mv dataset/CUB_200_2011 dataset/CUB
mv dataset/attributes.txt dataset/CUB/attributes/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EamOKGLoTuZdtcVYbHMWNpkn3iAVj8TP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EamOKGLoTuZdtcVYbHMWNpkn3iAVj8TP" -O dataset/segmentations.tgz && rm /tmp/cookies.txt
tar xvf dataset/segmentations.tgz -C dataset/ 
mv dataset/segmentations dataset/CUB
rm dataset/segmentations.tgz

wget -nc -O dataset/CUBV2.tar "https://onedrive.live.com/download?cid=B7111B95B80CCC66&resid=B7111B95B80CCC66%2130812&authkey=AFMzb4akufUiWU0"
mkdir dataset/CUBV2
tar xvf dataset/CUBV2.tar -C dataset/CUBV2
