#!/bin/bash

nohup psrecord 11016 --interval 30 --plot plot1.png --log log1.log > outputNohupScript.txt 2>&1 < /dev/null & echo $! > pid.txt