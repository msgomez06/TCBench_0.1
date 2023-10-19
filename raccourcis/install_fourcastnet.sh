conda activate ecmwf_venv/
git clone https://github.com/ecmwf-lab/ai-models-fourcastnetv2.git
cd ai-models-fourcastnetv2
pip3 install --upgrade -e .
cd ..
conda deactivate