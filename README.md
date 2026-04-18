# data_gen

Scripts for human-image data generation, including:

- balanced sampling of source image paths
- instruction generation with Qwen VL + vLLM
- target image generation with FireRed
- parquet filtering and mixing utilities

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Main scripts

- `balanced_sample_image_paths.py`: sample image paths evenly across immediate subdirectories
- `generate_instructions.py`: generate edit instructions and write resolution-sharded parquet outputs
- `generate_text_change_instructions.py`: generate text-change instructions following GEdit-Bench-style category ratios
- `generate_target_images.py`: generate edited target images with multi-GPU and multi-machine support
- `extract_parquet_fraction.py`: take a fraction of a parquet file
- `convert_instruction_parquet.py`: convert a legacy single instruction parquet into the new resolution-sharded format
- `mix_edit_parquets.py`: merge or mix parquet datasets
- `sample_image_paths.py`: sample image paths from a directory tree
- `firered_model_spec.py`, `firered_runtime.py`, `firered_fast_pipeline.py`: FireRed runtime helpers
- `test.py`: simple local inference test script

## Notes

- Large datasets, checkpoints, model artifacts, and generated outputs are intentionally not included in the repository.
- The current pipeline writes instruction outputs as resolution-sharded parquet files for faster downstream batching.
