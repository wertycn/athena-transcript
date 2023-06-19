import abc
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

from document_process import TranslateContext, DocumentProcessorFactory
from document_translator import DocumentTranslator


class FileListProvider(abc.ABC):
    """
    文件列表提供者抽象类
    """

    @abc.abstractmethod
    def get_file_list(self) -> List[str]:
        pass


class FileCollector(FileListProvider):
    """
    文件收集者，获取文件目录下所有文件的路径
    """

    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def get_file_list(self) -> List[str]:
        file_paths = [f for f in glob.glob(self.dir_path + "**/*", recursive=True) if os.path.isfile(f)]
        return file_paths


class FilesTranslationManager:
    """
    多文件翻译管理器，负责管理整个文件列表的翻译任务
    """

    def __init__(self, translator: DocumentTranslator, file_list_provider: FileListProvider):
        self.translator = translator
        self.file_list_provider = file_list_provider

    def process_all(self, source_dir, target_dir):
        file_list = self.file_list_provider.get_file_list()
        pool = ThreadPoolExecutor(max_workers=10)  # 指定线程池中的最大线程数

        def process_file(file):
            relative_path = os.path.relpath(file, source_dir)
            target_file = os.path.join(target_dir, relative_path)
            source_path = Path(file)
            context = TranslateContext(
                source_lang="English",
                target_lang="Chinese",
                source_path=source_path,
                target_path=Path(target_file),
            )

            DocumentProcessorFactory.create(source_path.suffix, self.translator, context).process_document()

        # 提交任务给线程池并异步执行
        futures = [pool.submit(process_file, file) for file in file_list]

        # 使用tqdm创建进度条，并根据任务的完成情况更新进度条
        with tqdm(total=len(futures), desc="Files Translate Processing") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

        print("All files processed.")


if __name__ == '__main__':
    provider = FileCollector("./")
    file_list = provider.get_file_list()
    print(file_list)
