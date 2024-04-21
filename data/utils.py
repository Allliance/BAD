import subprocess

def run_download_bash_file(script_path):
    subprocess.run(['bash', script_path])
    