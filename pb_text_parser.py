import json
from collections import OrderedDict


def str2builtin(v):
    try:
        temp = int(v)
        return temp
    except ValueError:
        pass
    try:
        temp = float(v)
        return temp
    except ValueError:
        pass
    if v in {'True', 'False'}:
        return v == 'True'
    return v.replace('\'', '').replace('"', '')


def parser_pb_txt(content):
    def process(split_char, no_check_key=False):
        has_key = False
        if not no_check_key:
            has_key = process_key(split_char)
        process_value(split_char, has_key)

    def process_key(split_char):
        # 根据stack判断是否为key, k 的后面不可能是v
        nonlocal stack
        if len(stack) > 0 and isinstance(item[-1], dict):
            k = str2builtin(''.join(stack))
            stack = []
            # 分隔符为: 只知道k，不知道类型
            if split_char == ':':
                value_type = 'none'
            # 分隔符为{ value的类型为dict
            elif split_char == '{':
                value_type = 'dict'
            # 分隔符为[ value的类型为list
            else:
                value_type = 'list'
            content_type.append((k, value_type))
            return True
        return False

    def process_value(split_char, has_key):
        if has_key:
            # 如果has_key 那么当前容器必然为一个dict，没有其他可能
            if content_type[-1][1] == 'none':
                # 此时stack必然为空，暂时无法得知value类型，跳过，等待下一次处理
                return
            elif content_type[-1][1] == 'dict':
                # 往当前dict放入内容
                v = OrderedDict()
                item[-1][content_type[-1][0]] = v
                # 由于当前放入了新的容器, item中需要指向新的容器
                item.append(v)
            else:
                v = []
                item[-1][content_type[-1][0]] = v
                # 由于当前放入了新的容器, item中需要指向新的容器
                item.append(v)
            return
        """
        来到这里，有几种情况：
        一、 上一次有key的情况下，类型为未知，在此处补上
        二、 来的是没有key的value，只可能是dict 或者 list
        三、 来得是 ] 或者 } 需要pop容器, 并且可能伴随str
        """
        nonlocal stack
        if len(content_type) > 0 and content_type[-1][1] == 'none':
            if split_char == '[':
                v = []
            elif split_char == '{':
                v = OrderedDict()
            else:
                # 如果为字符串则要对栈进行清空
                v = str2builtin(''.join(stack))
                stack = []
            item[-1][content_type[-1][0]] = v
            if isinstance(v, dict) or isinstance(v, list):
                item.append(v)
            content_type.pop()
            if split_char in {'}', ']'}:
                item.pop()
            return

        if split_char in {'{', '['}:
            v = OrderedDict() if split_char == '{' else []
            # 会走到这里的容器只有list一种可能
            item[-1].append(v)
            item.append(v)
            return

        if split_char in {'}', ']'}:
            # 伴随着字符串
            if len(stack) > 0:
                v = str2builtin(''.join(stack))
                stack = []
                if split_char == '}':
                    item[-1][content_type[-1][0]] = v
                    content_type.pop()
                if split_char == ']':
                    item[-1].append(v)
            # 弹出容器
            item.pop()
            return
        # 这种情况是由于list里存在,产生的
        if len(stack) > 0 and split_char == ',' and isinstance(item, list):
            v = str2builtin(''.join(stack))
            stack = []
            item[-1].append(v)

    # 根据 content 内容判断最外层容器类型
    if content[0] == '[':
        rst = []
    else:
        rst = OrderedDict()

    stack = []
    item = [rst]
    content_type = []
    for i, c in enumerate(content):
        if c == ' ':
            continue
        elif c == ':':
            process(c)
        elif c == '\n':
            process(c, no_check_key=True)
        elif c == '{':
            process(c)
        elif c == '[':
            process(c)
        elif c == '}':
            process(c, no_check_key=True)
        elif c == ']':
            process(c, no_check_key=True)
        elif c == ',':
            process(c, no_check_key=True)
        else:
            stack.append(c)
    # 去除最外层的临时数组
    if content[0] == '[':
        rst = rst[0]
    return rst


if __name__ == '__main__':
    test = '''
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
    '''
    print(json.dumps(parser_pb_txt(test), indent=2))