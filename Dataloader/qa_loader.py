import pickle as pkl
import json

file_path = './yago/GenTKG_data/test.pickle'
with open(file_path, 'rb') as f:
    data = pkl.load(f)
    print('successfully loaded')


def convert_pkl_to_json(pkl_file_paths, json_file_path):
    """
    从多个pkl文件读取数据，合并后转储为指定格式的json文件

    参数:
    pkl_file_paths (list): pkl文件路径列表
    json_file_path (str): json文件路径
    """
    all_items = []

    # 依次读取所有pkl文件
    for pkl_file in pkl_file_paths:
        try:
            with open(pkl_file, 'rb') as f:
                data = pkl.load(f)

                all_items.extend(data)
                print(f"已成功从 {pkl_file} 读取 {len(data)} 个项目")

        except FileNotFoundError:
            print(f"错误: 文件 {pkl_file} 不存在，跳过该文件")
        except Exception as e:
            print(f"错误: 处理文件 {pkl_file} 时发生异常 - {e}，跳过该文件")

    print(f"总共有 {len(all_items)} 个项目将被写入JSON文件")

    # 写入json文件
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            for item in all_items:
                # 确保每个字典都包含所需的键
                if 'question' not in item or 'answers' not in item:
                    print(f"警告: 发现格式不正确的项目: {item}")
                    continue

                # 创建符合要求的json行
                json_line = {
                    'question': item['question'],
                    'answer': ", ".join([f"{a}" for a in list(item['answers'])]),
                    'label': ", ".join([f"{a}" for a in list(item['answers'])])
                }

                # 写入单行json
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

        print(f"转换完成，已保存至: {json_file_path}")

    except Exception as e:
        print(f"错误: 写入JSON文件时发生异常 - {e}")

if __name__ == '__main__':
    data_path = './yago/GenTKG_data/'
    # input_files = ['train.pickle', 'valid.pickle', 'test.pickle']
    input_files = ['test.pickle']
    for i in range(len(input_files)):
        input_files[i] = data_path + input_files[i]
    output_file = './Data/yago/Question.json'
    convert_pkl_to_json(input_files, output_file)