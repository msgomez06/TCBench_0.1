source activate graphcast_venv/
git clone https://github.com/ecmwf-lab/ai-models-graphcast.git
cd ai-models-graphcast
pip3 install --upgrade -e .
pip3 install -r requirements-gpu.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd ..
conda deactivate