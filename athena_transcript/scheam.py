import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List



@dataclass
class TranslateContext:
    source_lang: str
    target_lang: str
    source_path: Path
    target_path: Path
    background: str = ""
    max_length: int = 500


class DocumentPiece:
    # 原始文本
    source_content = None
    # 转化后的内容
    replace_content = None
    # 翻译内容
    translate_content = None
    # 类型
    type = None
    # 是否需要翻译
    translate = True
    # 长度
    length = None
    # 元数据
    metadata = {}
    # 哈希值
    hash = None

    def __init__(self, source_content, piece_type,
                 replace_content="",
                 translate_content=None,
                 content_format=None,
                 translate=True,
                 metadata=None):
        self.source_content = source_content
        self.replace_content = replace_content
        self.translate_content = translate_content

        if translate and (translate_content is None or replace_content is None):
            self.translate_content = [source_content]
            self.replace_content = "<_%%0%%_>"

        self.type = piece_type
        self.format = content_format
        self.translate = translate
        self.length = len(source_content)
        self.metadata = metadata if metadata else {}

        self.hash = hashlib.md5(self.source_content.encode()).hexdigest()

    def to_dict(self):
        return {
            'type': self.type,
            'format': self.format,
            'translate': self.translate,
            'length': self.length,
            'source_content': self.source_content,
            'replace_content': self.replace_content,
            'translate_content': self.translate_content,
            'metadata': self.metadata,
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            source_content=data['source_content'],
            piece_type=data['type'],
            replace_content=data['replace_content'],
            translate_content=data['translate_content'],
            content_format=data['format'],
            translate=data['translate'],
            metadata=data['metadata']
        )


class TranscriptDocument:
    process_name: str
    version: str
    file_metadata: dict
    pieces: List[DocumentPiece]

    def __init__(self, process_name: str,  file_metadata: dict, pieces: List[DocumentPiece],version: str = "v1.0"):
        self.process_name = process_name
        self.version = version
        self.file_metadata = file_metadata
        self.pieces = pieces

    def to_dict(self):
        return {
            'process_name': self.process_name,
            'version': self.version,
            'file_metadata': self.file_metadata,
            'pieces': self.pieces
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: dict):
        pieces_data = data.pop('pieces')
        pieces = [DocumentPiece(**piece) for piece in pieces_data]
        return cls(pieces=pieces, **data)

