# BUSI-XAI

create a Python virtual environment via terminal:
python -m venv .venv

Then activate it:

PowerShell:
.\.venv\Scripts\Activate.ps1

Command Prompt (CMD):
.venv\Scripts\activate.bat

Once activated, you'll see (.venv) in your terminal.

Then upgrade pip and install packages from requirements.txt:

python -m pip install --upgrade pip
pip install -r requirements.txt

Then install packages from busi-xai:
pip install -e . 

To leave the environment:
deactivate