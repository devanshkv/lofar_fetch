#!/bin/bash

#'fil_file', 'snr', 'stime', 'dm', 'width',  'MJD_file', 'block_id', 'label', 'kill_mask_path'
#Time,DM,SNR,Width,MJD_file,tcand,log_width,members,block_id

while read cand
do
	snr=$(echo $cand | cut -d , -f 3)
	stime=$(echo $cand | cut -d , -f 6)
	dm=$(echo $cand | cut -d , -f 2)
	width=$(echo $cand | cut -d , -f 7)
	MJD_file=$(echo $cand | cut -d , -f 5)
	block_id=$(echo $cand | cut -d , -f 9)
	in_name=$(echo $1 | cut -d . -f 1 | cut -d_ -f 3 )
	fil_name=$(ls -1 *${in_name}*fil)
	echo "$fil_name,$snr,$stime,$dm,$width,$MJD_file,$block_id,,,"
done< <(cat $1 | tail -n +2)
