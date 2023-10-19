source activate graphcast_venv/
cd ai-models-graphcast && git pull && pip install --upgrade -e .
cd ..
conda deactivate