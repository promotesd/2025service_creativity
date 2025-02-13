import os

def parse_bio_file(file_path):
    """
    读取 BIO 格式文件：每行包含「字 标签」，
    空行表示一句/一段结束。
    
    将相连的实体标签 (B-XXX, I-XXX) 用 '【' 和 '】' 包裹起来，
    其余O标签直接输出到结果中。
    
    返回一个列表，每个元素对应一段合并后的文本字符串。
    """

    all_sentences = []  # 存放结果
    current_tokens = []  # 暂存当前段的文字

    in_entity = False  # 当前是否在实体区间
    entity_type = None  # 当前实体类型，如 KNOW/PRIN

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            # 空行 -> 一段结束
            if in_entity:
                # 如果还没关闭实体，就先关闭
                current_tokens.append('】')
                in_entity = False
                entity_type = None

            if current_tokens:
                # 拼接成完整字符串，存到结果
                all_sentences.append(''.join(current_tokens))
                current_tokens = []
            continue

        # 每行应为 "字 标签"
        parts = line.split()
        if len(parts) != 2:
            # 如果不符合格式，可选择跳过或处理
            continue

        token, label = parts
        label = label.upper()  # 转大写，避免大小写不一致

        if label.startswith('B-'):
            # 新实体开始
            if in_entity:
                # 先关闭旧实体
                current_tokens.append('】')
            in_entity = True
            entity_type = label[2:]  # B-KNOW -> KNOW
            current_tokens.append('【')
            current_tokens.append(token)
        elif label.startswith('I-'):
            # 同一实体延续
            if not in_entity:
                # 如果本来不在实体内，却来了 I-，也可以强行打开
                in_entity = True
                entity_type = label[2:]
                current_tokens.append('【')
            current_tokens.append(token)
        else:
            # label == 'O' 或其他
            if in_entity:
                # 结束当前实体
                current_tokens.append('】')
                in_entity = False
                entity_type = None
            # 普通字直接加入
            current_tokens.append(token)

    # 文档末尾，若还在实体内就关闭
    if in_entity:
        current_tokens.append('】')
        in_entity = False
        entity_type = None

    # 若最后一段没有空行结尾，也要收集
    if current_tokens:
        all_sentences.append(''.join(current_tokens))
        current_tokens = []

    return all_sentences


def main():
    """
    主函数：读取 train.txt (BIO格式)，
    解析并合并实体，然后将合并结果保存到指定目录下。
    """
    input_file = r"/root/autodl-tmp/dataset/mathbook/中学数学NER数据集/test.txt"
    output_dir = r"/root/autodl-tmp/code/2025service_creativity/process_dataset/mathNER"
    output_file = os.path.join(output_dir, "test.txt")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"输入文件不存在：{input_file}")
        return

    # 解析文件
    sentences = parse_bio_file(input_file)

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sent in enumerate(sentences):
            f.write(sent + "\n")
    
    print(f"处理完成！共合并 {len(sentences)} 段，结果已保存到：{output_file}")

if __name__ == "__main__":
    main()
