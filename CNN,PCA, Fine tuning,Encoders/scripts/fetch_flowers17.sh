#!/bin/bash


wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat
tar -xf 17flowers.tgz

rm 17flowers.tgz
rm datasplits.mat
rm -rf jpg
