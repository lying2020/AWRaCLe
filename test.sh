# Main table
CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/RESIDE/ --in_context_dir /data/data_awracle/Train/Dehaze/ --test_json dehaze_reside_test.json --output_path results/reside/ --in_context_file dehaze_reside_train.json --lpips --fid

CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/Rain13K/ --in_context_dir /data/data_awracle/Train/Derain/ --test_json derain_test_rain100l.json --output_path results/rain100l/ --in_context_file derain_train.json --lpips --fid

CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/Rain13K/ --in_context_dir /data/data_awracle/Train/Derain/ --test_json derain_test_rain100h.json --output_path results/rain100h/ --in_context_file derain_train.json --lpips --fid

CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/Snow100k/ --in_context_dir /data/data_awracle/Train/Desnow/ --test_json desnow_snow100_L_test.json --output_path results/snowL/ --in_context_file desnow_snow100_train.json --lpips --fid

CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/Snow100k/ --in_context_dir /data/data_awracle/Train/Desnow/ --test_json desnow_snow100_M_test.json --output_path results/snowM/ --in_context_file desnow_snow100_train.json --lpips --fid

CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/Snow100k/ --in_context_dir /data/data_awracle/Train/Desnow/ --test_json desnow_snow100_S_test.json --output_path results/snowS/ --in_context_file desnow_snow100_train.json --lpips --fid

# Mixed degradation on CSD: Snow + haze
CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/CSD/ --in_context_dir /data/data_awracle/Train/Desnow/ --test_json desnow_csd_Test.json --output_path results/csd_desnow/ --in_context_file desnow_snow100_train.json --lpips --fid

CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_name best.ckpt --test_dir /data/data_awracle/CSD/ --in_context_dir /data/data_awracle/Train/Dehaze/ --test_json desnow_csd_Test.json --output_path results/csd_dehaze/ --in_context_file dehaze_reside_train.json --lpips --fid