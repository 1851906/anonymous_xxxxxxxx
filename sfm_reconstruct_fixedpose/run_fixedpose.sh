for i in {1..13}; do
  echo "Scene $i | sparse COLMAP reconstruction"
  python fixedpose_sparse_colmap.py \
    --json ../data_RWAVS/${i}/transforms_train.json \
    --images-root ../data_RWAVS/${i}/frames \
    --out ../colmap_fixed/${i} \
    --subset-ratio 1.0 \
    --subset-strategy uniform \
    --matcher sequential \
    --overlap 10 \
    --use-gpu 1 \
    --copy-ply-to ../data_RWAVS/${i} \
    --copy-ply-name points3D.ply
done