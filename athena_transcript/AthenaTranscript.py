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

from athena_transcript.DocumentSpliter import DocumentSpliterFactory
from athena_transcript.DocumentTranslator import DocumentTranslator
from athena_transcript.scheam import TranscriptDocument, DocumentPiece

import logging

# 创建一个名为 'my_module' 的 logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


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


class AthenaTranscriptWorkspace:
    workspace_dir = '.athena_transcript'

    def __init__(self, initial_dir: Path):
        self.initial_dir = os.path.abspath(initial_dir)
        self.workspace = self.find_or_create_workspace()
        log.info(f"workspace is [{self.workspace}]")

    def find_or_create_workspace(self) -> str:
        current_dir = self.initial_dir

        # Check if workspace exists in the initial directory or its parents
        while True:
            workspace_path = os.path.join(current_dir, self.workspace_dir)
            if os.path.exists(workspace_path):
                return workspace_path
            else:
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # We've reached the root directory
                    break
                else:
                    current_dir = parent_dir

        # If workspace doesn't exist, create it in the initial directory
        workspace_path = os.path.join(self.initial_dir, self.workspace_dir)
        os.makedirs(workspace_path, exist_ok=True)
        #  创建工作空间的子目录
        return workspace_path

    def get_workspace(self):
        return self.workspace

    def get_record_path(self, hash: str, lange: str):
        return os.path.join(self.get_workspace(), "record", lange, hash)

    def is_exist_transcript_record(self, piece_hash: str, lange: str) -> bool:
        # Check if the file at the path obtained by get_record_path exists
        path = self.get_record_path(piece_hash, lange)
        return os.path.exists(path)

    def get_transcript_record(self, piece_hash: str, lange: str) -> str:
        path = self.get_record_path(piece_hash, lange)
        if os.path.exists(path):
            with open(path, 'r', encoding="utf-8") as f:
                return f.read()
        else:
            raise FileNotFoundError("The file does not exist.")

    def save_transcript_record(self, piece: DocumentPiece, lange: str):
        path = self.get_record_path(piece.hash, lange)
        with open(path, 'w', encoding="utf-8") as f:
            f.write(piece.translate_content)

class AthenaTranscript:
    default_excludes: List[str] = ["node_modules/", ".git/", ".idea/", AthenaTranscriptWorkspace.workspace_dir]

    def __init__(self, source_path: Path, target_path: Path, translator: DocumentTranslator,
                 target_lange: str = "English", source_lange: str = "Chinese",
                 excludes: str = None):
        log.info('Welcome to the AthenaTranscript document translation program!')
        self.workspace = AthenaTranscriptWorkspace(source_path)
        self.source_path = source_path
        self.target_lange = target_lange
        self.source_lange = source_lange
        if excludes is None:
            excludes = ""
        self.excludes = excludes
        # 获取所有文件列表
        exclude_list = excludes.split(",")
        exclude_list.append(target_path)
        exclude_list.extend([Path(value) for value in self.default_excludes])
        self.document_file_list = NormalFileCollector(source_path, exclude_list).get_file_list()
        # 获取需要翻译的文件列表
        self.translate_list, self.copy_list = self.filter_translate_file_list()
        self.translator = translator

    def 计算翻译成本(self):
        """
        基于遍历需要翻译的文件列表，转换为TranscriptDocument对象，调用Translator.计算成本 方法 来完成计算
        :return:
        """
        pass

    def predict_cost(self):
        # TODO: 调用分片处理函数，得到需要翻译的分片列表，再转换为分片实际翻译的文档 预处理对象
        token_num_list = []
        for path in self.translate_list:
            # 向分片工厂传入文件路径， 返回文档分片对象
            document = DocumentSpliterFactory.build_transcript_document(path)
            # 调用单分片计算方法
            token_num_list.append(self.translator.predict_cost(document))

        all_token = self.translator.sum_document_token(token_num_list)
        print(all_token)

        pass

    def predict_single_document_cost(self, document: TranscriptDocument):
        """
        预测单个文档的成本
        :return:
        """
        pieces = document.get_pieces()

        pass

    def translate(self):
        translate_list, copy_list = self.filter_translate_file_list()
        # 文件转换为TranscriptDocument 对象并存储

        # 计算本次翻译所需成本(如果有分片有翻译记录且无变化，不计算成本)

        # 不支持翻译的文件直接copy到目标目录

        # 执行翻译

        pass

    def __call__(self, *args, **kwargs):
        pass

    import os

    def filter_translate_file_list(self):
        """
        :return: a tuple of two lists, the first contains files that can be translated,
                 the second contains files that cannot be translated.
        """
        support_format = DocumentSpliterFactory.get_support_format()
        translate_list = []
        copy_list = []

        for file in self.document_file_list:
            _, ext = os.path.splitext(file)
            if ext == '':
                copy_list.append(file)
            elif ext[1:] in support_format:  # remove the dot from the extension
                translate_list.append(file)
            else:
                copy_list.append(file)

        return translate_list, copy_list


if __name__ == '__main__':
    transcript = AthenaTranscript(
        translator=DocumentTranslator(),
        source_path=Path("D:/mycode/weaviate-docs-zh-main"),
        target_path=Path("D:/mycode/weaviate-docs-zh-main-zh/"),
        excludes="*-cn.md,*_cn.md")
    print(transcript.translate_list)
    print(transcript.copy_list)

    transcript.predict_cost()
