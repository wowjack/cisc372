#!/bin/bash

str1=$(./reverse.exec hi friend)
test1=$(echo -e "ih\ndneirf")

str2=$(./reverse.exec test script test number two)
test2=$(echo -e "tset\ntpircs\ntset\nrebmun\nowt")

str3=$(./reverse.exec the f1nal reverse t)
test3=$(echo -e "eht\nlan1f\nesrever\nt")

if [ "$str1" == "$test1" ]; then
	echo "Test 1 passed"
else
	echo "Test 1 failed"
fi

if [ "$str2" == "$test2" ]; then
        echo "Test 2 passed"
else
        echo "Test 2 failed"
fi

if [ "$str3" == "$test3" ]; then
        echo "Test 3 passed"
else
        echo "Test 3 failed"
fi

