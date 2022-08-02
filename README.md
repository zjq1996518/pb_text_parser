由于公司使用triton部署模型，配置文件采用pbtxt格式，需要对配置文件进行一些动态的加载，没有找到很好的解析库，
所以自己写了一个。

因为就是一个小工具，就记录一下，也不打包了
调用方法：

```
from pb_text_parse import parser_pb_txt

file_path = ...
with open(file_path) as f:
    content = f.read()
rst = parser_pb_txt(content)
```
#例子：
## 待转换内容
config.pbtxt:
```json
name: "test"
backend: "python"
default_model_filename: "model.py"
max_batch_size: 0
input [
  {
    name: "input"
    data_type: TYPE_UINT8
    format: FORMAT_NHWC
    dims: [ 896, 896, 3 ]
  }
]
output [
  {
    name: "CLASSES"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "CONFIDENCES"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "BOXES"
    data_type: TYPE_INT32
    dims: [-1, 4]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0,1,2,3]
  }
]

parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/../model-py38.tar.gz"}
}
```
## 转换后：
为了方便展示，转换为了json,实际上都是python内置类型
```json
{
  "name": "test",
  "backend": "python",
  "default_model_filename": "model.py",
  "max_batch_size": 0,
  "input": [
    {
      "name": "input",
      "data_type": "TYPE_UINT8",
      "format": "FORMAT_NHWC",
      "dims": [
        896,
        896,
        3
      ]
    }
  ],
  "output": [
    {
      "name": "CLASSES",
      "data_type": "TYPE_INT32",
      "dims": [
        -1
      ]
    },
    {
      "name": "CONFIDENCES",
      "data_type": "TYPE_FP32",
      "dims": [
        -1
      ]
    },
    {
      "name": "BOXES",
      "data_type": "TYPE_INT32",
      "dims": [
        -1,
        4
      ]
    }
  ],
  "instance_group": [
    {
      "count": 1,
      "kind": "KIND_GPU",
      "gpus": [
        0,
        1,
        2,
        3
      ]
    }
  ],
  "parameters": {
    "key": "EXECUTION_ENV_PATH",
    "value": {
      "string_value": "$$TRITON_MODEL_DIRECTORY/../model-py38.tar.gz"
    }
  }
}
```