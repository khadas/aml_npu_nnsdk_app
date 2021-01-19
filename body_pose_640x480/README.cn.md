## 编译

opencv3

1. HOST:

```shell
$ ./build-cv3.sh <path to aml_npu_sdk/linux_sdk/linux_sdk >
```

2. LOCAL:

```shell
$ ./build-cv3.sh
```

## 运行

```shell
$ wget https://github.com/Amlogic-NN/AML_NN_SDK/raw/master/Model/DDK6.4.3/88/body_pose_88.nb
$ ./body_pose_640x480_picture ./body_pose_88.nb  < path to jpeg file>
```

VIM3L: 请使用  https://github.com/Amlogic-NN/AML_NN_SDK/raw/master/Model/DDK6.4.3/99/body_pose_99.nb

