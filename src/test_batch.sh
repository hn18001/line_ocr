# Usage: th ./test_run_batch.lua dev_id img_dir saved_result_file model_file
th ./test_run_batch.lua 2 ~/images/scene/save_img/ ./result.txt ../model/snapshot_170000.t7
python ../tools/edit_distance.py
