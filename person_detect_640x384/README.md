## Compile

opencv3

```shell
$ ./build-cv3.sh <path to aml_npu_sdk/linux_sdk/linux_sdk >
```

## Run

```shell
$ wget https://github.com/Amlogic-NN/AML_NN_SDK/raw/master/Model/DDK6.4.3/88/person_detect_88.nb
$ ./person_detect_640x384_picture ./person_detect_88.nb  < path to jpeg file>
$ ./person_detect_640x384_camera ./person_detect_88.nb  < path to camera node>
```

VIM3L: Please use https://github.com/Amlogic-NN/AML_NN_SDK/raw/master/Model/DDK6.4.3/99/person_detect_99.nb 
