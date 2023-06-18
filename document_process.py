import inspect
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from pathlib import Path
from typing import List

import nbformat

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
    def __init__(self, text, piece_type, metadata=None):
        self.text = text
        self.type = piece_type
        self.length = len(text)
        self.metadata = metadata if metadata else {}


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

    @staticmethod
    def create(document_format, translator, context):
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
            raise ValueError(f'No document processor for format {document_format}')

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

    def process_document(self):
        document = self.read_document(self.context.source_path)
        pieces = self.split_document(document, self.context.max_length)
        translated_pieces = self.translate_pieces(pieces)
        translated_document = self.combine_pieces(translated_pieces)
        self.save_document(translated_document)


class DocumentProcessorFactory:

    @staticmethod
    def create(document_format, translator, context) -> DocumentProcessor:
        return DocumentProcessor.create(document_format, translator, context)


class MarkdownProcessor(DocumentProcessor):

    @classmethod
    def get_support_format(cls) -> List[str]:
        return ['md', 'mdx', 'markdown']

    def read_document(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def split_document(self, document: str, max_length: int) -> List[DocumentPiece]:
        lines = document.split("\n")
        pieces = []
        buffer = ""
        in_code_block = False
        length = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("    "):  # Check for start or end of a code block
                if in_code_block:  # Code block ends
                    pieces.append(DocumentPiece(buffer, "code"))
                    buffer = ""
                    in_code_block = False
                else:  # Code block starts
                    if buffer:  # Add the previous buffered text as a piece
                        pieces.append(DocumentPiece(buffer, "text"))
                        buffer = ""
                    buffer += line  # Don't add newline if buffer is empty
                    in_code_block = True
            elif in_code_block:  # Inside a code block
                buffer += "\n" + line
            else:  # Normal text
                temp_len = length + len(line) + 1  # +1 for the newline character
                if temp_len > max_length:  # current line would exceed the max_length
                    pieces.append(DocumentPiece(buffer, "text"))
                    buffer = line  # Don't add newline if buffer is empty
                    length = len(line)
                else:
                    buffer += "\n" + line if buffer else line  # Don't add newline if buffer is empty
                    length = temp_len

        if buffer:  # Remaining text
            pieces.append(DocumentPiece(buffer, "text" if not in_code_block else "code"))

        return pieces

    def translate_pieces(self, pieces):
        translated_pieces = []
        for piece in pieces:
            if piece.type == "code":  # skip code blocks
                translated_pieces.append(piece)
            else:
                print("===============================================================")
                print("start translate piece:\n" + piece.text)

                translated_text = self.translator.translate(
                    piece.text,
                    target_language=self.context.target_lang,
                    document_format="Markdown"
                )
                print("translate result:\n" + translated_text)

                translated_pieces.append(DocumentPiece(translated_text, "text"))
        return translated_pieces

    def combine_pieces(self, pieces):

        return "\n".join((piece.text for piece in pieces))

    def save_document(self, document):
        with open(self.context.target_path, 'w', encoding='utf-8') as f:
            f.write(document)


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
    DocumentProcessorFactory.create("markdown", DocumentTranslator(), context).process_document()
