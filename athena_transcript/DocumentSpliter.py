import inspect
import json
import os
import re
import shutil
from abc import ABC, abstractmethod, ABCMeta
from datetime import datetime
from pathlib import Path
from typing import List

import frontmatter
# from frontmatter import Frontmatter

import nbformat
from tqdm import tqdm

from athena_transcript.DocumentTranslator import DocumentTranslator
from athena_transcript.scheam import TranslateContext, DocumentPiece, TranscriptDocument


class DocumentSpliterMeta(ABCMeta):
    processors = {}
    format_list = []

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        # 注册不是抽象类的类
        if not inspect.isabstract(cls):
            if not hasattr(cls, 'get_support_format'):
                raise NotImplementedError(f"Class {name} does not implement 'get_support_format' method.")
            support_format = cls.get_support_format()
            DocumentSpliterMeta.format_list.extend(support_format)
            DocumentSpliterMeta.processors.update({format: cls for format in support_format})


class DocumentSpliter(ABC, metaclass=DocumentSpliterMeta):

    def __init__(self,context, file_path: Path = None, max_length: int = 500):
        self.file_path = file_path
        self.max_length = max_length
        self.context = context

    @abstractmethod
    def read_document(self, filepath):
        pass

    @abstractmethod
    def split_document(self, content, max_length):
        pass

    @abstractmethod
    def translate_pieces(self, pieces):
        pass

    # def translate

    @abstractmethod
    def combine_pieces(self, pieces):
        pass

    @abstractmethod
    def save_document(self, document):
        """
        保存翻译结果到目标路径
        :param document:
        :return:
        """
        pass

    # def makedir_target_path(self):
    #     # 获取目录路径
    #     target_dir = os.path.dirname(self.context.target_path)
    #
    #     # 判断目录是否存在，不存在则创建
    #     if not os.path.exists(target_dir):
    #         os.makedirs(target_dir, exist_ok=True)

    @classmethod
    def get_support_format(cls) -> List[str]:
        """
        此分片支持的文件格式列表
        :return:
        """
        pass

    def to_translate_document(self) -> TranscriptDocument:
        pieces = self.split_document(self.read_document(self.file_path), self.max_length)

        extension = os.path.splitext(self.file_path)[1].lstrip('.')
        file_metadata = {
            "path": self.file_path,
            "format": extension,
            "name": os.path.basename(self.file_path),
            "extensions": extension,
            "last_change_time": datetime.fromtimestamp(os.path.getmtime(self.file_path)).isoformat()
        }

        translate_document = TranscriptDocument(
            process_name=type(self).__name__,
            version="1.0",
            file_metadata=file_metadata,
            pieces=pieces
        )

        return translate_document

    def print_pieces(self, pieces):
        piece_dict_list = [piece.to_dict() for piece in pieces]

        current_time = datetime.now()
        formatted_time = current_time.strftime("%y%m%d%H%M%S")
        directory = "./logs"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = self.context.target_path.name.replace('/', '__')
        log_file_name = f"{directory}/{formatted_time}_{file_name}.pieces.log.json"
        with open(log_file_name, "w", encoding='utf-8') as f:
            json.dump(piece_dict_list, f)

    def process_document(self):
        document = self.read_document(self.context.source_path)
        pieces = self.split_document(document, self.context.max_length)
        # self.print_pieces(pieces)
        translated_pieces = self.translate_pieces(pieces)
        translated_document = self.combine_pieces(translated_pieces)
        self.makedir_target_path()
        self.save_document(translated_document)


class DefaultSpliter(DocumentSpliter):

    @classmethod
    def get_support_format(cls) -> List[str]:
        """
        此分片支持的文件格式列表
        :return:
        """
        return []

    def read_document(self, filepath):
        pass

    def split_document(self, content, max_length):
        pass

    def translate_pieces(self, pieces):
        pass

    def combine_pieces(self, pieces):
        pass

    def save_document(self, document):
        #
        # 原地复制
        shutil.copy(self.context.source_path, self.context.target_path)

    def print_pieces(self, pieces):
        pass


class DocumentSpliterFactory:

    @staticmethod
    def create(path: Path, max_length: int = 500) -> DocumentSpliter:
        """
        创建文档处理器实例对象
        :param path:
        :param max_length:
        :return:
        """
        # 检查文件是否有后缀
        if not path.suffix:
            raise Exception("File does not have an extension")

        # 去掉"."获取文件后缀
        document_format = path.suffix[1:]

        # 检查处理器是否支持该格式
        if document_format in DocumentSpliterMeta.processors:
            return DocumentSpliterMeta.processors[document_format](path, max_length)
        else:
            return DefaultSpliter()
            # raise Exception(f"Not supported splitter format [{document_format}]")

    @staticmethod
    def build_transcript_document(path: Path, max_length=1000) -> TranscriptDocument:
        # 获取文件后缀
        # 基于后缀获取
        spliter = DocumentSpliterFactory.create(path, max_length)
        return spliter.to_translate_document()

    @staticmethod
    def get_support_format():
        return list(set(DocumentSpliterMeta.format_list))


class MarkdownSpliter(DocumentSpliter):
    """
    markdown 文档处理器
    """
    # TODO: frontmatter 作为一个分片添加到分片结果中去
    front_matter: any
    

    @classmethod
    def get_support_format(cls) -> List[str]:
        return ["md", "markdown", "mdx"]

    @classmethod
    def create_document_piece(cls, content, piece_type='text', translate=True, code_block_marker=None, **kwargs):
        if piece_type == 'code' and code_block_marker == ':::':
            # 提取代码块标题和内容
            findall = re.findall(r':::(\w+)?\s*(.*?)(?=\n:::)', content, re.DOTALL)
            if len(findall)<=0:
                return DocumentPiece(content, piece_type, translate=translate)

            code_block_title, content_piece = findall[0]
            content_piece = content_piece.strip()

            # 构建替换内容
            replace_content = f":::<_%%0%%_>\n<_%%1%%_>\n:::\n"

            # 构建元数据
            metadata = {
                'code_block_mark': ':::',
                'code_block_title': code_block_title
            }

            return DocumentPiece(content, replace_content, [code_block_title, content_piece], piece_type,
                                 translate=True, metadata=metadata)

        return DocumentPiece(content, piece_type, translate=translate)

    def read_document(self, filepath: str) -> str:
        with open(filepath, "r", encoding='utf-8') as file:
            document = file.read()
        matter = frontmatter.loads(document)
        self.front_matter = matter;
        return matter.content

    def split_document(self, content: str, max_length: int) -> List[DocumentPiece]:
        lines = content.split('\n')

        # Add newline to each line except the last one
        lines = [line + '\n' for line in lines[:-1]] + lines[-1:]

        pieces = []
        current_piece = ''
        in_code_block = False
        code_block_marker = None
        i = 0
        while i < len(lines):
            line = lines[i]

            # Toggle code block state
            if line.startswith('```') or line.startswith('~~~') or line.startswith(':::'):
                code_block_marker = line[:3]
                # If entering a code block, add previous piece as a text
                if not in_code_block and current_piece:
                    pieces.append(self.create_document_piece(current_piece))
                    current_piece = line
                    # just take the first 3 characters
                # If exiting a code block, add it as a piece
                elif in_code_block and current_piece and line is not None and line.strip().startswith(code_block_marker):
                    current_piece += line
                    pieces.append(self.create_document_piece(current_piece, 'code', translate=False,
                                                             code_block_marker=code_block_marker))
                    current_piece = ''
                    code_block_marker = None
                    continue  # prevent incrementing i
                in_code_block = not in_code_block
            # If in a code block, just add the line to the current piece
            elif in_code_block:
                current_piece += line
            # If the current piece plus the next line would exceed the max_length,
            # add the current_piece as a text piece, then start a new piece
            else:
                next_piece = current_piece + line if current_piece else line
                # if len(next_piece) > max_length:
                #     pieces.append(DocumentPiece(current_piece, 'text', translate=True))
                #     current_piece = line
                # else:
                current_piece = next_piece
            i += 1

        # Add the remaining text as a piece
        if current_piece:
            pieces.append(
                DocumentPiece(current_piece, 'text' if not in_code_block else 'code', translate=not in_code_block))

        #  处理文本分片内可能存在JSX 相关标记, 重新分片
        pieces = self.do_mdx(pieces)
        #  基于分片长度， 重新分片
        pieces = self.do_pieces_length(pieces)

        return pieces

    def translate_pieces(self, pieces):
        # Implementation depends on DocumentTranslator, translate the text pieces
        translated_pieces = []

        for piece in tqdm(pieces, desc=f"{self.context.source_path} translate processing"):
            if piece.type == 'text':
                translated_text = self.translator.translate(
                    piece.text, target_language=self.context.target_lang, document_format="Markdown",
                    background=self.context.background
                )
                translated_pieces.append(DocumentPiece(translated_text, "text"))
            else:
                translated_pieces.append(piece)

        return translated_pieces

    def combine_pieces(self, pieces) -> str:
        return "".join([piece.text for piece in pieces])

    def save_document(self, document: str):
        with open(self.context.target_path, 'w', encoding='utf-8') as file:
            self.front_matter.content = document
            file.write(frontmatter.dumps(self.front_matter))

    @classmethod
    def do_mdx(cls, pieces: List[DocumentPiece]):
        result = []
        for piece in pieces:
            if piece.type == 'text':
                result.append(None)
            else:
                result.append(piece)
        return pieces

    @classmethod
    def do_pieces_length(cls, pieces):
        return pieces


class SimpleTextSpliter(DocumentSpliter):

    @classmethod
    def get_support_format(cls) -> List[str]:
        return []

    def read_document(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def split_document(self, content: str, max_length: int) -> List[DocumentPiece]:
        lines = content.split("\n")
        pieces = []
        buffer = ""
        length = 0

        for line in lines:
            temp_len = length + len(line) + 1  # +1 for the newline character
            if temp_len > max_length:  # current line would exceed the max_length
                pieces.append(DocumentPiece(buffer, "text"))
                buffer = line  # Don't add newline if buffer is empty
                length = len(line)
            else:
                buffer += "\n" + line if buffer else line  # Don't add newline if buffer is empty
                length = temp_len

        if buffer:  # Remaining text
            pieces.append(DocumentPiece(buffer, "text"))

        return pieces

    def translate_pieces(self, pieces):
        translated_pieces = []
        for piece in pieces:
            # print("===============================================================")
            # print("start translate piece:\n" + piece.text)

            translated_text = self.translator.translate(
                piece.text
            )
            # print("translate result:\n" + translated_text)

            translated_pieces.append(DocumentPiece(translated_text, "text"))
        return translated_pieces

    def combine_pieces(self, pieces):
        return "\n".join((piece.text for piece in pieces))

    def save_document(self, document):
        with open(self.context.target_path, 'w', encoding='utf-8') as f:
            f.write(document)


class NotebookSpliter(DocumentSpliter):

    @classmethod
    def get_support_format(cls) -> List[str]:
        # 暂不启用
        return ['ipynbxxx', 'notebookxx']

    def read_document(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def split_document(self, content: str, max_length: int) -> List[DocumentPiece]:
        notebook = nbformat.reads(content, as_version=4)
        pieces = []

        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                pieces.extend(self._split_markdown_cell(cell, max_length))
            elif cell.cell_type == "code":
                pieces.append(DocumentPiece(cell.source, "code", {"cell_type": cell.cell_type}))
            else:
                pieces.append(DocumentPiece(cell.source, "unknown", {"cell_type": cell.cell_type}))

        return pieces

    def _split_markdown_cell(self, cell, max_length):
        # Reuse the MarkdownProcessor's logic to split Markdown text
        markdown_processor = MarkdownSpliter(self.translator, self.context)
        return markdown_processor.split_document(cell.source, max_length)

    def translate_pieces(self, pieces):
        translated_pieces = []
        for piece in pieces:
            if piece.type == "text":
                translated_text = self.translator.translate(
                    piece.text,
                    target_language=self.context.target_lang,
                    document_format="markdown"
                )
                translated_pieces.append(DocumentPiece(translated_text, "text"))
            else:  # skip code blocks
                translated_pieces.append(piece)
        return translated_pieces

    def combine_pieces(self, pieces):
        translated_notebook = nbformat.v4.new_notebook()
        for piece in pieces:
            if piece.type in ["code", "unknown"]:
                cell = nbformat.v4.new_code_cell(piece.text)
            else:
                cell = nbformat.v4.new_markdown_cell(piece.text)
            cell.cell_type = piece.metadata.get("cell_type", cell.cell_type)
            translated_notebook.cells.append(cell)

        return nbformat.writes(translated_notebook)

    def save_document(self, document):
        notebook = nbformat.reads(document, as_version=4)
        with open(self.context.target_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)


if __name__ == '__main__':
    # 配置日志记录
    context = TranslateContext(
        "English", "Chinese",
        Path("../tests/sample/markdown/frontmatter/yaml.md"),
        Path("../tests/sample/markdown/frontmatter/yaml_cn.md"),
    )
    processor = MarkdownSpliter("../tests/sample/markdown/mdx/index.md")
    document = processor.to_translate_document()

    print(document.to_dict())
    # DocumentProcessorFactory.create("md", DocumentTranslator(), context).process_document()
    # context = TranslateContext(
    #     "English", "Chinese",
    #     Path("../tests/sample/markdown/frontmatter/yaml-long-content.md"),
    #     Path("../tests/sample/markdown/frontmatter/yaml-long-content-cn.md"),
    # )
    # DocumentProcessorFactory.create("md", DocumentTranslator(), context).process_document()
