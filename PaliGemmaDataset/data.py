import struct
import os
import json
import re # 导入正则表达式模块
import numpy as np
from PIL import Image, ImageDraw

class WPTTParser:
    """
    用于解析 CASIA-OLHWDB 数据库中 .wptt 文件的类。

    功能：
    - 为每个 .wptt 文件生成一个包含完整笔迹的 PNG 图像。
    - 将时间、X坐标、Y坐标信息编码到图像的 RGB 通道中。
    """
    def __init__(self, file_path):
        """
        初始化解析器。

        Args:
            file_path (str): .wptt 文件的完整路径。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        self.file_path = file_path
        self.header = {}
        self.strokes = []
        self.lines = []

    def _read_and_unpack(self, f, fmt, size):
        buffer = f.read(size)
        if len(buffer) < size:
            raise IOError(f"尝试读取 {size} 字节失败，文件可能已损坏或提前结束。")
        return struct.unpack(f'<{fmt}', buffer)[0]

    def _read_string(self, f, size):
        buffer = f.read(size)
        return buffer.split(b'\0', 1)[0].decode('ascii')

    def parse(self):
        """
        执行文件解析，包括文件头、笔划和行数据。
        """
        with open(self.file_path, 'rb') as f:
            self._parse_header(f)
            self._parse_strokes(f)
            self._parse_lines(f)

    def _parse_header(self, f):
        self.header['head_size'] = self._read_and_unpack(f, 'I', 4)
        self.header['format_code'] = self._read_string(f, 8)
        illustration_len = self.header['head_size'] - 54
        if illustration_len > 0:
            self.header['illustration'] = f.read(illustration_len).strip(b'\0').decode('gbk', 'ignore')
        else:
            self.header['illustration'] = ""
        self.header['code_type'] = self._read_string(f, 20)
        self.header['code_length'] = self._read_and_unpack(f, 'H', 2)
        self.header['data_type'] = self._read_string(f, 20)
        self.header['sample_length'] = self._read_and_unpack(f, 'I', 4)
        self.header['page_index'] = self._read_and_unpack(f, 'I', 4)
        self.header['stroke_num'] = self._read_and_unpack(f, 'I', 4)

    def _parse_strokes(self, f):
        total_strokes = self.header['stroke_num']
        for _ in range(total_strokes):
            point_num = self._read_and_unpack(f, 'H', 2)
            stroke_points = []
            for _ in range(point_num):
                x, y = struct.unpack('<HH', f.read(4))
                stroke_points.append({'x': x , 'y': y })
            self.strokes.append(stroke_points)

    def _parse_lines(self, f):
        try:
            line_num = self._read_and_unpack(f, 'H', 2)
        except IOError:
            return
        for i in range(line_num):
            line_stroke_num = self._read_and_unpack(f, 'H', 2)
            line_stroke_indices = [self._read_and_unpack(f, 'H', 2) for _ in range(line_stroke_num)]
            line_char_num = self._read_and_unpack(f, 'H', 2)
            line_label = ''
            for _ in range(line_char_num):
                tag_code_bytes = f.read(self.header['code_length'])
                if tag_code_bytes == b'\xff\xff': continue
                line_label += tag_code_bytes.decode('gbk', 'ignore')
            self.lines.append({'line_index': i, 'stroke_indices': line_stroke_indices, 'text': line_label})

    def create_page_image_and_get_info(self, output_dir="output", image_size=448):
        """
        为整个文件（页面）创建一个编码图像并返回信息。
        - R: 时间/顺序 (t / t_max)
        - G: X坐标 ((x - x_min) / (x_max - x_min))
        - B: Y坐标 ((y - y_min) / (y_max - y_min))
        
        Returns:
            tuple: (生成的图像文件名, 对应的完整文本) 或 (None, None) 如果失败。
        """
        all_points = [point for stroke in self.strokes for point in stroke]
        if not all_points:
            print(f"警告: 文件 {os.path.basename(self.file_path)} 中没有笔划数据，已跳过。")
            return None, None
        
        # 将所有行的文本合并
        raw_text = "".join([line['text'] for line in self.lines])
        
        # 【修正】: 使用正则表达式处理 \u0000
        # 1. 删除所有出现在数字和括号（半角和全角）后的 \u0000
        processed_text = re.sub(r'([\d()\uff08\uff09])\u0000', r'\1', raw_text)
        # 2. 将剩余的所有 \u0000 替换为换行符
        full_text = processed_text.replace('\u0000', '\n')

        points_np = np.array([[p['x'], p['y']] for p in all_points])
        xmin, ymin = np.min(points_np, axis=0)
        xmax, ymax = np.max(points_np, axis=0)

        page_width = xmax - xmin if xmax > xmin else 1
        page_height = ymax - ymin if ymax > ymin else 1
        total_points = len(all_points)

        img = Image.new('RGB', (image_size, image_size), 'white')
        draw = ImageDraw.Draw(img)

        scale = min((image_size - 20) / page_width, (image_size - 20) / page_height)
        offset_x = (image_size - page_width * scale) / 2
        offset_y = (image_size - page_height * scale) / 2
        
        dynamic_width = max(1, round(image_size / 896.0))

        global_point_idx = 0
        for stroke in self.strokes:
            if len(stroke) < 2:
                global_point_idx += len(stroke)
                continue

            for i in range(len(stroke) - 1):
                p1 = stroke[i]
                p2 = stroke[i+1]

                r = int(255-(global_point_idx / (total_points-1)) * 255) if total_points > 1 else 0
                g = int(((p1['x'] - xmin) / page_width) * 255)
                b = int(((p1['y'] - ymin) / page_height) * 255)
                color = (r, g, b)

                x1 = (p1['x'] - xmin) * scale + offset_x
                y1 = (p1['y'] - ymin) * scale + offset_y
                x2 = (p2['x'] - xmin) * scale + offset_x
                y2 = (p2['y'] - ymin) * scale + offset_y

                draw.line([(x1, y1), (x2, y2)], fill=color, width=dynamic_width)
                global_point_idx += 1
            
            global_point_idx += 1

        base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        image_filename = f"{base_filename}.png"
        image_path = os.path.join(output_dir, image_filename)
        img.save(image_path)
        
        return image_filename, full_text


# --- 主程序入口 ---
if __name__ == '__main__':
    try:
        file_to_parse = "data/WPTT2.0-Train/003-P16.wptt"
        output_directory = "encoded_page_images"

        parser = WPTTParser(file_to_parse)
        parser.parse()
        
        parser.create_page_image_and_get_info(output_dir=output_directory, image_size=896)

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保 'file_to_parse' 变量指向一个有效的 .wptt 文件。")
    except Exception as e:
        print(f"\n处理过程中发生未知错误: {e}")



