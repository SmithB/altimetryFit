#! /usr/bin/env bash
#
# Download a PGC 2-m strip DEM (with its matchtag and bitmask), and filter it
# to coarser resolutions with dem_filter.py
#
# usage: get_pgc_2res.sh URL [--res R1 [R2 ...]]
#   URL    the strip's .json metadata URL
#   --res  the output resolutions, in m (default: 32 16).  For each resolution
#          R, the dem_filter.py arguments are read from
#          ../filter_defaults_${R}m.txt, and the filtered DEM is written to ${R}m/.
#
# Resolutions whose filtered DEM already exists are skipped (and if all of them
# exist, nothing is downloaded).  The raw 2-m files are deleted only when the
# filtered DEMs for all of the resolutions exist.  Exits with status 1 if a
# defaults file is missing (before downloading anything), if a download fails,
# or if a filter step fails; status 2 for bad arguments.

prog=$(basename $0)

usage() {
    echo "usage: $prog URL [--res R1 [R2 ...]]   (resolutions in m, default: 32 16)" >&2
    exit 2
}

url=""
res_list=()
while [ $# -gt 0 ]; do
    case $1 in
        --res)
            shift
            n_res=0
            while [ $# -gt 0 ] && [[ $1 =~ ^[0-9]+m?$ ]]; do
                res_list+=(${1%m})
                n_res=$((n_res+1))
                shift
            done
            [ $n_res -gt 0 ] || { echo "$prog: --res needs at least one resolution" >&2; usage; }
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "$prog: unknown option $1" >&2
            usage
            ;;
        *)
            [ -z "$url" ] || { echo "$prog: more than one URL ($url, $1)" >&2; usage; }
            url=$1
            shift
            ;;
    esac
done
[ -n "$url" ] || usage
[ ${#res_list[@]} -gt 0 ] || res_list=(32 16)

# every resolution needs its defaults file
missing=0
for res in "${res_list[@]}"; do
    defaults_file=../filter_defaults_${res}m.txt
    if [ ! -f $defaults_file ]; then
        echo "$prog: $defaults_file not found (needed for the ${res}m DEM)" >&2
        missing=1
    fi
done
[ $missing -eq 0 ] || exit 1

dem_url=$(echo $url | sed 's/\.json$/_dem.tif/')
match_url=$(echo $url | sed 's/\.json$/_matchtag.tif/')
mask_url=$(echo $url | sed 's/\.json$/_bitmask.tif/')
dem_file=$(basename $dem_url)
match_file=$(basename $match_url)
mask_file=$(basename $mask_url)

filt_file_for() {
    # filtered DEM for resolution $1
    echo ${1}m/$(echo $dem_file | sed s/_2m_/_${1}m_/ | sed s/_dem.tif/_dem_filt.tif/)
}

echo "dem_url="$dem_url
echo "dem_file="$dem_file

# resolutions still to do
todo=()
for res in "${res_list[@]}"; do
    [ -f $(filt_file_for $res) ] || todo+=($res)
done

if [ ${#todo[@]} -gt 0 ]; then
    # download to a temporary name, so that an interrupted download is not
    # mistaken for a complete one
    for this_url in $url $dem_url $match_url $mask_url; do
        file=$(basename $this_url)
        [ -f $file ] && continue
        wget -O $file.part $this_url 2>&1 | grep -o "[0-9.]\+ [KM]*B/s"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "$prog: could not download $this_url" >&2
            rm -f $file.part
            exit 1
        fi
        mv $file.part $file
    done

    for res in "${todo[@]}"; do
        filt_file=$(filt_file_for $res)
        defaults_file=../filter_defaults_${res}m.txt
        mkdir -p ${res}m
        echo "filt_file="$filt_file", defaults_file="$defaults_file
        if ! dem_filter.py $dem_file $filt_file @${defaults_file} || [ ! -f $filt_file ]; then
            echo "$prog: the ${res}m filter step failed for $dem_file" >&2
            # a partial output would be taken as done by the next run
            rm -f $filt_file
        fi
    done
fi

for res in "${res_list[@]}"; do
    if [ ! -f $(filt_file_for $res) ]; then
        echo "$prog: no ${res}m DEM for $dem_file, keeping the raw files" >&2
        exit 1
    fi
done
rm -f $dem_file $match_file $mask_file
