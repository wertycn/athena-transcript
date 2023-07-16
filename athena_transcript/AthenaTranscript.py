"""
AthenaTranscript 翻译程序入口

调用方法
   AthenaTranscript(document_dir = "with translate docs dir",source_lange="English",target_lange="Chinese",excludes="")

参数含义：

"""
import abc
import glob
import os
from pathlib import Path
from typing import List

import pathspec


class FileListProvider(abc.ABC):
    """
    文件列表提供者抽象类
    """

    def __init__(self, path: Path, excludes: List[str]):
        path = path.as_posix()
        if not path.endswith('/'):
            path += '/'
        self.path = path
        self.excludes = excludes
        # Create a pathspec object using the .gitignore style and the exclude list
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, excludes)

        # Get all files in the directory
        all_files = [f for f in Path(self.path).rglob('*') if f.is_file()]

        # Filter files according to exclude list
        self.file_list = [f for f in all_files if not spec.match_file(f.as_posix())]

    @abc.abstractmethod
    def get_file_list(self) -> List[Path]:
        return self.file_list


class NormalFileCollector(FileListProvider):

    def get_file_list(self) -> List[str]:
        return self.file_list


class AthenaTranscript:

    def __init__(self, source_path: Path, target_path: Path,
                 target_lange: str = "English", source_lange: str = "Chinese",
                 excludes: str = None):
        self.source_path = source_path
        self.target_lange = target_lange
        self.source_lange = source_lange
        if excludes is None:
            excludes = ""
        self.excludes = excludes
        # 获取所有文件列表
        self.document_file_list = NormalFileCollector(source_path, excludes.split(",")).get_file_list()

    def translate(self):
        # 获取需要翻译的文件列表

        # 文件转换为TranscriptDocument 对象并存储

        # 计算本次翻译所需成本(如果有分片有翻译记录且无变化，不计算成本)

        # 不支持翻译的文件直接copy到目标目录

        # 执行翻译

        pass

    def __call__(self, *args, **kwargs):
        pass
