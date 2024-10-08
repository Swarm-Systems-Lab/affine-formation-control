import os

file_path = os.path.dirname(__file__)
deps_path = os.path.join(file_path, "requirements.txt")

os.system("pip install -r " + deps_path)
os.system(
    "pip install git+https://github.com/Swarm-Systems-Lab/ssl_pysimutils.git@main"
)
