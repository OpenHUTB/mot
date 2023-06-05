# python service of similarity computation
cd /data2/whd/workspace/MOT/TKP
nohup /home/d/anaconda2/envs/deepMOT/bin/python /data2/whd/workspace/MOT/TKP/calculateSimilarityTKP.py >/data2/whd/workspace/MOT/TKP/log/calculateSimilarityTKP.log 2>&1 &


# wait python server launch totallly
sleep 8

# matlab service of tracking
# matlab -nodisplay
# Add an "mrun" alias for running matlab in the terminal.
# alias mrun="matlab -nodesktop -nosplash -logfile `date +%Y_%m_%d-%H_%M_%S`.log -r"
nohup matlab -nosplash -nodesktop -r 'cd /data2/whd/workspace/MOT/TKP;DMAN_demo;quit;' >/data2/whd/workspace/MOT/TKP/log/DMAN_demo.log 2>&1 &
# matlab -r 'cd /data2/whd/workspace/MOT/TKP;DMAN_demo;'


# clear process
# ps aux | grep calculateSimilarityTKP.py | awk '{print $2}' | xargs kill -s 9
