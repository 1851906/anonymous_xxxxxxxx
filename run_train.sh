DATA_DIR="./data_RWAVS"
LOG_DIR="./logs_S2A_NVAS"

for r in $(seq 1.0 0.1 1.0)
do
  RSTR=$(printf "%.1f" "$r")
  echo "==== Subset ratio: ${RSTR} ===="

  for i in {1..13}; do
    echo "Room $i  | subset_ratio=${RSTR}"
    python main.py \
      --data-root "${DATA_DIR}/${i}/" \
      --log-dir "${LOG_DIR}/subset_${RSTR}/" \
      --output-dir "${i}/" \
      --subset_ratio "${RSTR}" \
      --room_number "${i}" \
      --use_ori \
      --use_boundary_token \
      --lr 5e-4 \
      --batch-size 6 \
      --max-epoch 100 \
      --save-freq 20 \
      --device "cuda:0"
  done
done
