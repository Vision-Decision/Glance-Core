import subprocess
subprocess.Popen(['bash','-c','bash /content/amb/setup_run.sh > /content/amb/setup.log 2>&1'], start_new_session=True)
print('started')
