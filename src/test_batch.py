import os

iteration = 10000
iteration_end = 38000

while iteration <= iteration_end:
    cmd = "th ./test_run_batch.lua 2 ~/images/scene/save_img/ ./result.txt snapshot_" # scene test
    #cmd = "th ./test_run_batch.lua 2 ~/images/news_caption/Test_Img/ ./result.txt snapshot_" # news caption
    cmd = cmd + str(iteration) + ".t7"
    print cmd
    os.system(cmd)
    os.system("python ../tools/edit_distance.py")
    iteration = iteration + 2000
