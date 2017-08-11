#!/bin/bash

mydir=$(pwd)
iam=shu
echo "Is your GSA id: $iam ? "
echo "If not $iam , please edit 0cp2ccc.sh "

echo "cp $mydir to syncdir..."

mails=$(echo $mydir | tr "/" "\n")

for addr in $mails
do
    thisdir=$addr
done

echo rsync -avz --progress --exclude '*.pyc' --exclude-from '0rsync_exclude' $mydir $iam@dccxl001.pok.ibm.com:/u/$iam/syncdir/

rsync -avz --progress --exclude '*.pyc' --exclude-from '0rsync_exclude' $mydir $iam@dccxl001.pok.ibm.com:/u/$iam/syncdir/

echo "cp to ccc done!"