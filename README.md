# altimetryFit
Scripts for fitting smooth surfaces to altimetry data


This repository includes code that uses the utilities in the smithB/LSsurf and smithB/pointCollection repos to fit smoothly changing surfaces to altimetry data from Greenland and Antarctica.

A sample workflow for this repository is:

### 1. Download altimetry files:

Supported formats include:
    
    - ICESat hdf5
    - ICESat-2 ATL06, ATL11 
    - Reduced ATM Qfit (see scripts/reduce_ATM.py)
    - Reduced LVIS
    - Reduced PGC DEMs (see scripts/dem_filter.py)
    - Cryosat-2 level-2
    - Retracked Cryosat-2 level-1


Each datatype should be stored in its own directory

### 2.  Use the pointCollection index_glob.py script to create a geoIndex file for each altimetry-file directory

### 3.  Create a json file containing a list of geoIndex files:

    The json file should translate to a Python dictionary, in which the keys are the data types, and  the values are the corresponding geoIndex filenames, or wildcards that reduce to a list of geoIndex filenames

Example (default_args/Arctic_GI_files_IS2_era.json) :
```
{
    "ICESat2": "/Volumes/ice1/ben/ATL06/tiles/Arctic/005/*/GeoIndex.h5",
    "DEM": "/Volumes/insar10/ben/ArcticDEM/v2*/*/GeoIndex.h5"
}

```
### 4. Create a default_args file containing arguments to fit_altimetry.py

Example ()

### 5. Generate tiled preliminary fits

The preliminary fits produce independent tiles of fit data (e.g. 30x30-km tiles, with centers spaced 20 km apart).  These can be mosaicked directly, or if distracting discontinuities are visible between tiles, step 6 allows matching at tile boundaries.

scripts/make_OIB_queue.py with the --prelim argument generates commands to make tiles that span the area defined by the --mask_file argument in the default_args file, in steps of the --tile_spacing argument

Run the commands in the preliminary queue, with xargs or using scripts in the smithB/parallel_boss repo.

### 6. Rerun the tiles, with forced edge matching

Run scripts/make_OIB_queue.py with the --matched argument, and run that list of commands.

### 7. Mosaic the output.

scripts/regen_OIB_mosaics will generate mosaics of the dz and z0 fields from the tiles.


