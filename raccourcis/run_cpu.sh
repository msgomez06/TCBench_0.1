conda activate ecmwf_venv/
pip uninstall onnxruntime-gpu
pip uninstall onnxruntime
pip uninstall ai-models-panguweather && pip install onnxruntime
cd ai-models-panguweather && pip install .
cd ..
ai-models --assets ./panguweather/ --input cds --date 20180110 --time 0000 --lead-time 30 --path './panguweather/pangu_{date}_{step}h.grib' panguweather
pip uninstall ai-models-panguweather && pip install onnxruntime-gpu
cd ai-models-panguweather && ONNXRUNTIME=onnxruntime-gpu pip install .
cd ..
conda deactivate