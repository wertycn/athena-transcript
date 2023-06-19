import inspect
import json
import os
import shutil
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from datetime import time, datetime
from pathlib import Path
from typing import List

import nbformat
from tqdm import tqdm

from document_translator import DocumentTranslator


@dataclass
class TranslateContext:
    source_lang: str
    target_lang: str
    source_path: Path
    target_path: Path
    background: str = ""
    max_length: int = 1000


class DocumentPiece:
    def __init__(self, text, piece_type, translate=True, metadata=None):
        self.text = text
        self.type = piece_type
        self.translate = translate
        self.length = len(text)
        self.metadata = metadata if metadata else {}

    def to_dict(self):
        return {
            'text': self.text,
            'type': self.type,
            'translate': self.translate,
            'length': self.length,
            'metadata': self.metadata
        }


class DocumentProcessorMeta(ABCMeta):
    processors = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        # 注册不是抽象类的类
        if not inspect.isabstract(cls):
            if not hasattr(cls, 'get_support_format'):
                raise NotImplementedError(f"Class {name} does not implement 'get_support_format' method.")
            support_format = cls.get_support_format()
            DocumentProcessorMeta.processors.update({format: cls for format in support_format})


class DocumentProcessor(ABC, metaclass=DocumentProcessorMeta):

    def __init__(self, translator: DocumentTranslator, context: TranslateContext):
        self.translator = translator
        self.context = context

    @abstractmethod
    def read_document(self, filepath):
        pass

    @abstractmethod
    def split_document(self, document, max_length):
        pass

    @abstractmethod
    def translate_pieces(self, pieces):
        pass

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

    @classmethod
    def get_support_format(cls) -> List[str]:
        """
        此分片支持的文件格式列表
        :return:
        """
        pass

    def print_pieces(self, pieces):
        piece_dict_list = [piece.to_dict() for piece in pieces]

        current_time = datetime.now()
        formatted_time = current_time.strftime("%y%m%d%H%M%S")
        directory = "./logs"
        if not os.path.exists(directory):
            os.makedirs(directory)
        log_file_name = f"{directory}/{formatted_time}_{self.context.target_path.replace('/', '__')}.pieces.log.json"
        with open(log_file_name, "w", encoding='utf-8') as f:
            json.dump(piece_dict_list, f)

    def process_document(self):
        document = self.read_document(self.context.source_path)
        pieces = self.split_document(document, self.context.max_length)
        self.print_pieces(pieces)
        translated_pieces = self.translate_pieces(pieces)
        translated_document = self.combine_pieces(translated_pieces)
        self.save_document(translated_document)


class DefaultProcessor(DocumentProcessor):

    @classmethod
    def get_support_format(cls) -> List[str]:
        """
        此分片支持的文件格式列表
        :return:
        """
        return []
    def read_document(self, filepath):
        pass

    def split_document(self, document, max_length):
        pass

    def translate_pieces(self, pieces):
        pass

    def combine_pieces(self, pieces):
        pass

    def save_document(self, document):
        # 原地复制
        shutil.copy(self.context.source_path, self.context.target_path)


class DocumentProcessorFactory:

    @staticmethod
    def create(document_format: str, translator: DocumentTranslator, context: TranslateContext) -> DocumentProcessor:
        """
        创建文档处理器实例对象
        :param document_format:
        :param translator:
        :param context:
        :return:
        """
        if document_format in DocumentProcessorMeta.processors:
            return DocumentProcessorMeta.processors[document_format](translator, context)
        else:
            return DefaultProcessor(translator, context)


class MarkdownProcessor(DocumentProcessor):
    def __init__(self, translator: DocumentTranslator, context: TranslateContext):
        super().__init__(translator, context)
        self.max_length = context.max_length

    @classmethod
    def get_support_format(cls) -> List[str]:
        return ["md", "markdown"]

    def read_document(self, filepath: str) -> str:
        with open(filepath, "r", encoding='utf-8') as file:
            document = file.read()
        return document

    def split_document(self, document: str, max_length: int) -> List[DocumentPiece]:
        lines = document.split('\n')

        # Add newline to each line except the last one
        lines = [line + '\n' for line in lines[:-1]] + lines[-1:]

        pieces = []
        current_piece = ''
        in_code_block = False
        i = 0
        while i < len(lines):
            line = lines[i]

            # Toggle code block state
            if line.startswith('```'):
                # If entering a code block, add previous piece as a text
                if not in_code_block and current_piece:
                    pieces.append(DocumentPiece(current_piece, 'text', translate=True))
                    current_piece = line
                # If exiting a code block, add it as a piece
                elif in_code_block and current_piece:
                    current_piece += line
                    pieces.append(DocumentPiece(current_piece, 'code', translate=False))
                    current_piece = ''
                    continue  # prevent incrementing i
                in_code_block = not in_code_block
            # If in a code block, just add the line to the current piece
            elif in_code_block:
                current_piece += line
            # If the current piece plus the next line would exceed the max_length,
            # add the current_piece as a text piece, then start a new piece
            else:
                next_piece = current_piece + line if current_piece else line
                if len(next_piece) > max_length:
                    pieces.append(DocumentPiece(current_piece, 'text', translate=True))
                    current_piece = line
                else:
                    current_piece = next_piece
            i += 1

        # Add the remaining text as a piece
        if current_piece:
            pieces.append(
                DocumentPiece(current_piece, 'text' if not in_code_block else 'code', translate=not in_code_block))

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
            file.write(document)


class SimpleTextProcessor(DocumentProcessor):

    @classmethod
    def get_support_format(cls) -> List[str]:
        return ['txt', 'text']

    def read_document(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def split_document(self, document: str, max_length: int) -> List[DocumentPiece]:
        lines = document.split("\n")
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
            print("===============================================================")
            print("start translate piece:\n" + piece.text)

            translated_text = self.translator.translate(
                piece.text
            )
            print("translate result:\n" + translated_text)

            translated_pieces.append(DocumentPiece(translated_text, "text"))
        return translated_pieces

    def combine_pieces(self, pieces):
        return "\n".join((piece.text for piece in pieces))

    def save_document(self, document):
        with open(self.context.target_path, 'w', encoding='utf-8') as f:
            f.write(document)


class NotebookProcessor(DocumentProcessor):

    @classmethod
    def get_support_format(cls) -> List[str]:
        return ['ipynb', 'notebook']

    def read_document(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def split_document(self, document: str, max_length: int) -> List[DocumentPiece]:
        notebook = nbformat.reads(document, as_version=4)
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
        markdown_processor = MarkdownProcessor(self.translator, self.context)
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
        "tests/sample/markdown/getting_started.md",
        "tests/sample/markdown/getting_started_cn.md",
    )
    DocumentProcessorFactory.create("md", DocumentTranslator(), context).process_document()
