conda activate ecmwf_venv/
cd ai-models-fourcastnetv2 && git pull && pip install --upgrade -e .
cd ..
conda deactivate