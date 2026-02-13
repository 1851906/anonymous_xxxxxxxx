#
for r in $(seq 1.0 0.1 1.0)
do
  RSTR=$(printf "%.1f" "$r")
  echo "==== Subset ratio: ${RSTR} ===="
    for i in {1..13}; do
    python eval.py \
    --data-root ./data_RWAVS/${i}/ \
    --ckpt logs_S2A_NVAS/subset_${RSTR}/${i}/99.pth \
    --device cuda \
    --batch-size 1 \
    --num-workers 4 \
    --room_number "${i}" \
    --use_ori \
    --use_boundary_token
  done
done


