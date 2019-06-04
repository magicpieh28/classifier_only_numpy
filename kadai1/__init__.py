import platform
import sys
import socket
from pathlib import Path

REQUIRED_PYTHON_VERSION = (3, 7)

if sys.version_info < REQUIRED_PYTHON_VERSION:
    exit(f'Python 3.7 or higher version is required')

host_name: str = socket.gethostname()
system_name: str = platform.system().lower()

app_dir = Path(__file__).expanduser().absolute().parent
project_dir = app_dir.parent
project_data_dir = project_dir / 'data'

kadai1_dir = project_data_dir / 'retrieva-intern2019' / '1'
