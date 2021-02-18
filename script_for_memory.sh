#!/bin/bash

nohup psrecord 12824 --interval 1 --plot plot1.png --log log1.log > outputNohupScript.txt 2>&1 < /dev/null & echo $! > pid.txt