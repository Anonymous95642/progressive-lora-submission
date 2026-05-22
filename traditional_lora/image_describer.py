"""
图像描述服务
提供高级图像描述功能和批量处理能力

"""

import os
import time
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
import logging
from model_loader import LLaVAModelLoader, create_model_loader

logger = logging.getLogger(__name__)

class ImageDescriber:
    def __init__(self, 
                 model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False):
        """
        初始化图像描述服务
        
        Args:
            model_path: 模型路径
            load_in_8bit: 是否使用8bit量化
            load_in_4bit: 是否使用4bit量化
        """
        self.model_path = model_path
        self.model_loader = None
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # 预定义的描述模板
        self.prompt_templates = {
            "detailed": "请详细描述这张图片的内容，包括场景、人物、物体、颜色、构图等所有可见元素。",
            "simple": "简单描述这张图片的主要内容。",
            "scene": "描述这张图片中的场景和环境。",
            "objects": "列出并描述这张图片中的主要物体。",
            "people": "描述这张图片中的人物及其动作。",
            "artistic": "从艺术角度分析这张图片的构图、色彩和风格。",
            "mood": "描述这张图片传达的情感和氛围。"
        }
    
    def initialize(self) -> bool:
        """
        初始化模型
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("初始化图像描述服务...")
            self.model_loader = create_model_loader(
                self.model_path,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit
            )
            logger.info("图像描述服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            return False
    
    def describe_single_image(self, 
                            image_path: str,
                            prompt_type: str = "detailed",
                            custom_prompt: Optional[str] = None,
                            max_tokens: int = 512,
                            temperature: float = 0.7) -> Dict:
        """
        描述单张图像
        
        Args:
            image_path: 图像路径
            prompt_type: 提示类型 (detailed, simple, scene, objects, people, artistic, mood)
            custom_prompt: 自定义提示（优先于prompt_type）
            max_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            Dict: 包含描述结果的字典
        """
        if self.model_loader is None:
            raise RuntimeError("模型未初始化，请先调用initialize()")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            # 确定使用的提示
            if custom_prompt:
                prompt = custom_prompt
            elif prompt_type in self.prompt_templates:
                prompt = self.prompt_templates[prompt_type]
            else:
                prompt = self.prompt_templates["detailed"]
            
            start_time = time.time()
            
            # 生成描述
            description = self.model_loader.describe_image(
                image=image_path,  # 传递图像路径
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            end_time = time.time()
            
            # 获取图像信息
            image = Image.open(image_path)
            
            result = {
                "image_path": image_path,
                "image_size": image.size,
                "prompt_type": prompt_type,
                "prompt": prompt,
                "description": description,
                "generation_time": round(end_time - start_time, 2),
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            }
            
            logger.info(f"成功描述图像: {os.path.basename(image_path)} (耗时: {result['generation_time']}秒)")
            return result
            
        except Exception as e:
            logger.error(f"描述图像失败 {image_path}: {str(e)}")
            raise
    
    def describe_batch_images(self, 
                            image_paths: List[str],
                            prompt_type: str = "detailed",
                            custom_prompt: Optional[str] = None,
                            max_tokens: int = 512,
                            temperature: float = 0.7,
                            save_results: bool = True,
                            output_file: str = "image_descriptions.txt") -> List[Dict]:
        """
        批量描述多张图像
        
        Args:
            image_paths: 图像路径列表
            prompt_type: 提示类型
            custom_prompt: 自定义提示
            max_tokens: 最大生成token数
            temperature: 生成温度
            save_results: 是否保存结果到文件
            output_file: 输出文件名
            
        Returns:
            List[Dict]: 描述结果列表
        """
        if self.model_loader is None:
            raise RuntimeError("模型未初始化，请先调用initialize()")
        
        results = []
        total_images = len(image_paths)
        
        logger.info(f"开始批量处理 {total_images} 张图像...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                logger.info(f"处理第 {i}/{total_images} 张图像: {os.path.basename(image_path)}")
                
                result = self.describe_single_image(
                    image_path=image_path,
                    prompt_type=prompt_type,
                    custom_prompt=custom_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"处理图像失败 {image_path}: {str(e)}")
                # 添加错误结果
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        # 保存结果
        if save_results:
            self._save_results_to_file(results, output_file)
        
        successful_count = len([r for r in results if "error" not in r])
        logger.info(f"批量处理完成: {successful_count}/{total_images} 张图像成功处理")
        
        return results
    
    def describe_directory(self, 
                          directory_path: str,
                          image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
                          **kwargs) -> List[Dict]:
        """
        描述目录中的所有图像
        
        Args:
            directory_path: 目录路径
            image_extensions: 支持的图像文件扩展名
            **kwargs: 传递给describe_batch_images的其他参数
            
        Returns:
            List[Dict]: 描述结果列表
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        # 查找图像文件
        image_paths = []
        for ext in image_extensions:
            pattern = f"*{ext}"
            image_paths.extend(Path(directory_path).glob(pattern))
            image_paths.extend(Path(directory_path).glob(pattern.upper()))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            logger.warning(f"在目录 {directory_path} 中未找到图像文件")
            return []
        
        logger.info(f"在目录 {directory_path} 中找到 {len(image_paths)} 张图像")
        
        return self.describe_batch_images(image_paths, **kwargs)
    
    def _save_results_to_file(self, results: List[Dict], output_file: str):
        """保存结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"图像描述结果 - 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, result in enumerate(results, 1):
                    f.write(f"图像 {i}: {os.path.basename(result.get('image_path', 'Unknown'))}\n")
                    f.write("-" * 50 + "\n")
                    
                    if "error" in result:
                        f.write(f"错误: {result['error']}\n")
                    else:
                        f.write(f"图像尺寸: {result.get('image_size', 'Unknown')}\n")
                        f.write(f"提示类型: {result.get('prompt_type', 'Unknown')}\n")
                        f.write(f"生成时间: {result.get('generation_time', 'Unknown')}秒\n")
                        f.write(f"描述: {result.get('description', 'No description')}\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
            
            logger.info(f"结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存结果文件失败: {str(e)}")
    
    def cleanup(self):
        """清理资源"""
        if self.model_loader:
            self.model_loader.cleanup()
            self.model_loader = None

# 便捷函数
def quick_describe(image_path: str, 
                  model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                  prompt_type: str = "detailed") -> str:
    """
    快速描述单张图像的便捷函数
    
    Args:
        image_path: 图像路径
        model_path: 模型路径
        prompt_type: 提示类型
        
    Returns:
        str: 图像描述
    """
    describer = ImageDescriber(model_path)
    try:
        if not describer.initialize():
            raise RuntimeError("模型初始化失败")
        
        result = describer.describe_single_image(image_path, prompt_type)
        return result["description"]
    finally:
        describer.cleanup()

