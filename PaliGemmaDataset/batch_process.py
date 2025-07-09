import os
import json
from data import WPTTParser 
from tqdm import tqdm

def process_all_folders(root_data_dir, root_output_dir, image_size=448):
    """
    批量处理指定根目录下的所有子文件夹中的 .wptt 文件。

    Args:
        root_data_dir (str): 包含多个子文件夹的根数据目录 (例如 'data')。
        root_output_dir (str): 用于存放所有结果的根输出目录 (例如 'output')。
        image_size (int): 生成图像的尺寸。
    """
    # 检查根数据目录是否存在
    if not os.path.isdir(root_data_dir):
        print(f"错误: 数据目录 '{root_data_dir}' 不存在。")
        return

    # 遍历根数据目录下的每一个子文件夹 (如 WPTT2.0-Train, WPTT2.0-Test 等)
    for subdir_name in sorted(os.listdir(root_data_dir)):
        source_subdir = os.path.join(root_data_dir, subdir_name)
        
        # 确保它是一个目录
        if not os.path.isdir(source_subdir):
            continue

        print(f"\n--- 正在处理文件夹: {subdir_name} ---")

        # 创建对应的输出子文件夹
        output_subdir = os.path.join(root_output_dir, subdir_name)
        os.makedirs(output_subdir, exist_ok=True)

        # 收集该子文件夹下所有的 .wptt 文件
        wptt_files = [f for f in os.listdir(source_subdir) if f.endswith('.wptt')]
        if not wptt_files:
            print(f"在 {source_subdir} 中没有找到 .wptt 文件。")
            continue
            
        # 初始化用于存储该文件夹所有信息的元数据字典
        metadata_for_subdir = {}
        
        # 使用 tqdm 创建进度条
        for filename in tqdm(wptt_files, desc=f"处理 {subdir_name}"):
            file_path = os.path.join(source_subdir, filename)
            
            try:
                # 1. 创建解析器实例并解析文件
                parser = WPTTParser(file_path)
                parser.parse()
                
                # 2. 创建图像并获取文件名和文本信息
                img_filename, text = parser.create_page_image_and_get_info(
                    output_dir=output_subdir, 
                    image_size=image_size
                )
                
                # 3. 如果成功生成，则添加到元数据中
                if img_filename and text is not None:
                    metadata_for_subdir[img_filename] = text
                    
            except Exception as e:
                print(f"\n处理文件 {file_path} 时发生错误: {e}")

        # 4. 处理完一个子文件夹的所有文件后，写入该文件夹的 metadata.json
        if metadata_for_subdir:
            json_path = os.path.join(output_subdir, 'metadata.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_for_subdir, f, ensure_ascii=False, indent=4)
            print(f"--- {subdir_name} 的元数据已保存至 {json_path} ---")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 设置你的数据根目录和输出根目录
    DATA_ROOT = "data"  # 包含 WPTT2.0-Train 等文件夹的目录
    OUTPUT_ROOT = "encoded_dataset" # 所有处理结果的存放目录
    IMAGE_SIZE = 896
    
    print("开始批量处理...")
    process_all_folders(DATA_ROOT, OUTPUT_ROOT, IMAGE_SIZE)
    print("\n所有文件夹处理完毕！")

