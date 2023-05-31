#! /usr/bin/env bash

url=$1

dem_url=`echo $url | sed s/.json/_dem.tif/`
match_url=`echo $url | sed s/.json/_matchtag.tif/`
mask_url=`echo $url | sed s/.json/_bitmask.tif/`
dem_file=`basename $dem_url`
filt_file=$(echo $dem_file | sed s/_2m_/_32m_/ | sed s/_dem.tif/_dem_filt.tif/)

echo "dem_url="$dem_url
echo "dem_file="$dem_file
echo "filt_file="$filt_file

match_file=$(basename $match_url)
mask_file=$(basename $mask_url)

for url in $dem_url $match_url $mask_url; do
    file=$(basename $url)
    [ -f $file ] || wget -q $url
done

#wget -q $dem_url && wget -q $match_url && wget -q $mask_url || $(echo "COULD NOT DOWNLOAD"; exit 1)
 
[ -f default_args.txt ] && dem_filter.py $dem_file $filt_file @default_args.txt

#[ -f $filt_file ] && $(rm $dem_file $match_file $mask_file)

