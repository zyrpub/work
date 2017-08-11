#!/bin/bash

mydir=$(pwd)
iam=srallap

echo "Is your GSA id: $iam ? "
echo "If not $iam , please edit 0cp2.sh "

echo "cp $mydir to syncdir..."

mails=$(echo $mydir | tr "/" "\n")

for addr in $mails
do
    thisdir=$addr
done

echo rsync -avz --progress --exclude '*.pyc' --exclude-from '0rsync_exclude' $mydir $iam@dccxl001.pok.ibm.com:/u/$iam/eeg/syncdir/

rsync -avz --progress --exclude '*.pyc' --exclude-from '0rsync_exclude' $mydir $iam@dccxl001.pok.ibm.com:/u/$iam/eeg/syncdir/

echo "cp to ccc done!"