#!/bin/bash

echo "1. running knn..."
cd cuda/nn
# ./nn filelist_4 -r 5 -lat 30 -lng 90
sh run
cd ../..
echo "\n"

echo "2. running h3d..."
cd cuda/hotspot3D
# ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out
sh run
cd ../..
echo "\n"

echo "3. running lud..."
cd cuda/lud
# cuda/lud_cuda -s 4096 -v
sh run
cd ../..
echo "\n"

echo "4. running pf..."
cd cuda/pathfinder
# ./pathfinder 1000000 100 20
sh run
cd ../..
echo "\n"

echo "5. running gs..."
cd cuda/gaussian
# ./gaussian -s 12000
sh run
cd ../..
echo "\n"

echo "6. running lmd..."
cd cuda/lavaMD
# ./lavaMD -boxes1d 120
sh run
cd ../..
echo "\n"

echo "test finished!"

