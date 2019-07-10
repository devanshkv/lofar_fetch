# lofar_fetch
FETCH bindings for the LOFAR data

`cluster_dat.py` clusters the `.dat` file using dbscan. See `cluster_dat.py -h` to check and tweak the clustering parameters.

`dbs2cand_csv.sh` makes a csv file which `candmaker_lofar.py` can read. Needs the `.dbs` file created by the `cluster_dat.py`. Usage: `dbs2cand_csv.sh XYZ.dbs`

`candmaker_lofar.py` makes the `.h5` files needed by `predict.py` (from [fetch](https://github.com/devanshkv/fetch)). See `candmaker_lofar.py -h` for details.
