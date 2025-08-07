```python
import os
import glob
import numpy as np
from PIL import Image
import nibabel as nib
from tqdm import tqdm


def sorted_by_number(file_list):
    """按文件名中的数字序号排序（适用于类似 'tongue_P0986_1.jpg' 的命名）"""

    def extract_num(path):
        filename = os.path.basename(path)
        # 提取文件名中的数字部分（如 'P0986_1' 中的 1）
        num_str = filename.split('_')[-1].split('.')[0]  # 分割后取最后一个数字部分
        return int(num_str) if num_str.isdigit() else 0  # 非数字返回0（可根据实际情况调整）

    return sorted(file_list, key=extract_num)


def process_jpg_to_mask(jpg_path, target_shape):
    """将JPG图像转换为二值掩码数组，并调整尺寸以匹配目标形状（高×宽）"""
    try:
        # 1. 读取图像并转换为灰度图
        img = Image.open(jpg_path).convert('L')  # 'L'表示灰度模式

        # 2. 顺时针旋转90度（expand=True避免裁剪边缘）
        img_rotated = img.rotate(-90, expand=True)

        # 3. 水平翻转（左右镜像）
        img_flipped = img_rotated.transpose(Image.FLIP_LEFT_RIGHT)

        # 4. 调整尺寸以匹配NII切片的实际尺寸（高×宽）
        target_height, target_width = target_shape
        # 注意：PIL.resize的参数是 (width, height)，因此需要交换目标高宽
        resized_img = img_flipped.resize((target_width, target_height))

        # 5. 二值化处理：大于0的像素置为1，其余为0
        mask_array = (np.array(resized_img) > 0).astype(np.uint8)

        return mask_array
    except Exception as e:
        print(f"⚠️ 处理JPG失败 {jpg_path}: {str(e)}")
        return None


def save_nii_mask(save_path, mask_array, template_nii):
    """将掩码数组保存为NIfTI格式文件（使用模板的空间信息，适配通道维度）"""
    try:
        # 获取模板NII的形状（可能包含通道维度）
        template_shape = template_nii.shape

        # 调整掩码形状以匹配模板（关键修复：添加通道维度）
        # 模板形状可能是 (高, 宽, 1) 或 (高, 宽)，掩码需与之完全一致
        if len(template_shape) == 3 and template_shape[2] == 1:
            # 模板是3D（高, 宽, 1），掩码需扩展为3D（高, 宽, 1）
            if len(mask_array.shape) == 2:
                mask_array = mask_array[..., np.newaxis]  # 添加通道维度
        elif len(template_shape) == 2:
            # 模板是2D（高, 宽），掩码需保持2D
            if len(mask_array.shape) != 2:
                raise ValueError(f"掩码维度不匹配（模板2D，掩码{len(mask_array.shape)}D）")
        else:
            raise ValueError(f"不支持的模板维度 {template_shape}")

        # 创建新的NIfTI图像（使用模板的仿射矩阵和头信息）
        nii_img = nib.Nifti1Image(mask_array, template_nii.affine, template_nii.header)
        nib.save(nii_img, save_path)
        return True
    except Exception as e:
        print(f"❌ 保存NII失败 {save_path}: {str(e)}")
        return False


def main():
    base_dir = r"F:\graduate\tongue\xinxueguan1397"
    data_dir = os.path.join(base_dir, "data1")  # JPG文件夹
    images_dir = os.path.join(base_dir, "images1")  # NII文件夹（单通道2D切片）
    masks_dir = os.path.join(base_dir, "masks")  # 输出文件夹

    # 创建输出目录（不存在则自动创建）
    os.makedirs(masks_dir, exist_ok=True)

    # 获取并排序文件（按数字序号严格对应）
    jpg_files = sorted_by_number(glob.glob(os.path.join(data_dir, "*.jpg")))
    nii_files = sorted_by_number(glob.glob(os.path.join(images_dir, "*.nii.gz")))

    # 打印文件数量（关键调试）
    print(f"📂 JPG文件数量: {len(jpg_files)}")
    print(f"📂 NII文件数量: {len(nii_files)}")

    # 校验文件数量一致性（必须一一对应）
    if len(jpg_files) != len(nii_files):
        raise ValueError(f"❌ JPG文件数量({len(jpg_files)})与NII文件数量({len(nii_files)})不匹配")

    # 处理所有文件对（每个JPG对应一个NII切片）
    with tqdm(total=len(jpg_files), desc="处理进度") as pbar:
        for idx in range(len(jpg_files)):
            jpg_path = jpg_files[idx]
            nii_path = nii_files[idx]
            filename = os.path.splitext(os.path.basename(jpg_path))[0]

            # 步骤1：加载当前NII文件并获取实际尺寸（高×宽×通道）
            try:
                template_nii = nib.load(nii_path)  # 加载当前NII文件
            except Exception as e:
                print(f"❌ 加载NII文件失败 {nii_path}: {str(e)}")
                pbar.update(1)
                continue

            nii_shape = template_nii.shape
            print(f"\n🔸 处理文件 {filename}（索引{idx}/{len(jpg_files) - 1}）:")
            print(f"   NII文件形状: {nii_shape}")

            # 步骤2：处理JPG并调整尺寸至NII切片的实际尺寸（高×宽）
            mask_array = process_jpg_to_mask(jpg_path, target_shape=nii_shape[:2])  # 目标尺寸为高×宽
            if mask_array is None:
                pbar.update(1)
                continue
            print(f"   JPG转换后掩码形状（未适配通道）: {mask_array.shape}")

            # 步骤3：适配掩码形状以匹配NII文件的维度（关键修复）
            try:
                # 根据NII的维度调整掩码的通道维度
                if len(nii_shape) == 3 and nii_shape[2] == 1:
                    # NII是3D（高, 宽, 1），掩码需扩展为3D（高, 宽, 1）
                    if mask_array.ndim == 2:
                        mask_array = mask_array[..., np.newaxis]  # 添加通道维度
                    elif mask_array.ndim != 3:
                        raise ValueError(f"掩码维度错误（需2D或3D，实际{mask_array.ndim}D）")
                elif len(nii_shape) == 2:
                    # NII是2D（高, 宽），掩码需保持2D
                    if mask_array.ndim != 2:
                        raise ValueError(f"掩码维度错误（需2D，实际{mask_array.ndim}D）")
                else:
                    raise ValueError(f"不支持的NII维度 {nii_shape}")

                # 验证适配后的掩码形状与NII形状完全一致
                if mask_array.shape != nii_shape:
                    raise ValueError(f"掩码形状不匹配（NII: {nii_shape} vs 掩码: {mask_array.shape}）")

                print(f"   适配后掩码形状（匹配NII）: {mask_array.shape}")
            except Exception as e:
                print(f"❌ 形状适配失败 {filename}: {str(e)}")
                pbar.update(1)
                continue

            # 步骤4：保存为NII文件（使用当前NII的空间信息）
            nii_save_path = os.path.join(masks_dir, f"{filename}.nii.gz")
            if save_nii_mask(nii_save_path, mask_array, template_nii):
                pbar.set_postfix_str(f"✅ 保存成功: {filename}")
            else:
                pbar.update(1)

    print("\n🎉 所有文件处理完成！")


if __name__ == "__main__":
    main()
```

