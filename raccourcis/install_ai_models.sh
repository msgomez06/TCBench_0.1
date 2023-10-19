conda activate ecmwf_venv/
git clone https://github.com/ecmwf-lab/ai-models.git
cd ai-models
pip3 install --upgrade -e .
cd ..
conda deactivate