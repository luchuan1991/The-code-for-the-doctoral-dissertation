#提取hsv rgb,lab



    import os
    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from skimage import io, color  # 用于Lab转换


    def extract_color_features(image_path):
        """提取图像的9个颜色通道特征（Lab, HSV, RGB）"""
        # 读取图像（OpenCV默认BGR顺序）
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            return None

        # 转换为RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 转换为Lab颜色空间
        lab_img = color.rgb2lab(rgb_img)  # 使用D65白点

        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # 分离通道
        l_channel, a_channel, b_channel = cv2.split(lab_img.astype(np.float32))
        h_channel, s_channel, v_channel = cv2.split(hsv_img)
        r_channel, g_channel, b_channel_rgb = cv2.split(rgb_img)

        # 计算每个通道的均值
        features = {
            'L_mean': np.mean(l_channel),
            'a_mean': np.mean(a_channel),
            'b_mean': np.mean(b_channel),
            'H_mean': np.mean(h_channel),
            'S_mean': np.mean(s_channel),
            'V_mean': np.mean(v_channel),
            'R_mean': np.mean(r_channel),
            'G_mean': np.mean(g_channel),
            'B_mean': np.mean(b_channel_rgb)
        }

        return features


    def process_tongue_images(folder_path):
        """处理文件夹中的所有舌象图像"""
        # 获取所有JPG文件
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

        results = []
        for filename in tqdm(image_files, desc="处理舌象图像"):
            img_path = os.path.join(folder_path, filename)

            # 提取特征
            features = extract_color_features(img_path)
            if features is None:
                print(f"警告: 无法读取图像 {filename}")
                continue

            # 添加文件名ID
            image_id = os.path.splitext(filename)[0]
            result = {'ID': image_id, **features}
            results.append(result)

        # 转换为DataFrame并保存
        df = pd.DataFrame(results)
        csv_path = os.path.join(folder_path, 'tongue_color_features.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')

        return df, csv_path


    if __name__ == "__main__":
        # 配置路径
        folder_path = r"F:\graduate\tongue\xinxueguan1397\data"

        # 处理图像并保存结果
        result_df, csv_path = process_tongue_images(folder_path)

        # 打印结果
        print(f"\n✅ 处理完成! 共分析了 {len(result_df)} 张舌象图像")
        print(f"📊 结果已保存至: {csv_path}")
        print("\n前5个样本的统计特征:")
        print(result_df.head())

