# UncertaintyNet
Repository for my Masters thesis!!!

# Installation
1. `virtualenv -p "/usr/bin/python3.8" venv`
2. `source venv/bin/activate` 
3. `python -m pip install --upgrade pip`
4. `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
5. `pip install -r requirements.txt`

# Installation with conda
1. `conda create -n myenv`
2. `conda install pip`
3. `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia`
4. `pip install -r requirements.txt`