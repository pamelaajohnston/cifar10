# QPNet
A CNN to measure the quantisation parameters. Also, a bunch of different (selectable) network architectures. The commandline for it goes something like this:

For training:
`CUDA_VISIBLE_DEVICES=$GPU0 python qpNet/qpNet_multi_gpu_train.py --train_dir /home/1609098/dev/local/qp_train --data_dir /home/1609098/dev/AllVid_80 --eval_dir /home/1609098/dev/local/eval --mylog_dir /home/1609098/dev/local --mylog_dir_eval . --max_steps 100000 --network_architecture $NUM --binarise_label -2 >> test.txt`

For evaluation:
`CUDA_VISIBLE_DEVICES=$GPU0 nohup python qpNet/qpNet_eval.py --checkpoint_dir /home/1609098/dev/local/qp_train --data_dir /home/1609098/dev/AllVid_80 --eval_dir /home/1609098/dev/local/eval --mylog_dir_eval . --num_examples 8400 --run_times 3 --network_architecture $NUM --binarise_label -2`

Note that binarise_label is to do with the creation of super-classes (which is done as the dataset is read into the network rather than manipulating the dataset itself).

This code (untidy though it is) was used in the paper [Johnston, Elyan, and Jayne. “Toward video tampering exposure: Inferring compression parameters from pixels” in 19th International Conference on Engineering Applications of Neural Networks (EANN), Springer, 2018](https://doi.org/10.1007/978-3-319-98204-5_4).

