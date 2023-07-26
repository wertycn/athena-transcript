import argparse
import os
import re
from typing import List

import yaml
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

from athena_transcript.scheam import TranscriptDocument


class DocumentTranslator:
    few_shot_example: list[BaseMessage]

    separator = "[*-*-*-*]"

    def __init__(self, llm: BaseChatModel = None, config_path: str = None):
        self.llm = self.build_model(llm)
        self.config = self.load_config(config_path)
        self.default_language = "Chinese"
        self.default_format = "Markdown"
        self.few_shot_example = self.build_chat_sequence()

    @staticmethod
    def build_model(llm):
        if llm is None:
            model = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
            llm = ChatOpenAI(model=model)

        return llm

    @staticmethod
    def load_config(path):
        if path is None:
            dir_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(dir_path, '../prompt.yaml')

        with open(path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    @staticmethod
    def extract_whitespace(text):
        start_whitespace = ""
        end_whitespace = ""
        valid_text = text.strip()

        start_match = re.match(r"^(\s*)", text)
        if start_match:
            start_whitespace = start_match.group(1)

        end_match = re.search(r"(\s*)$", text)
        if end_match:
            end_whitespace = end_match.group(1)

        return start_whitespace, valid_text, end_whitespace

    def translate(self, text, target_language=None, document_format=None, background=None, **kwargs):
        # 如果没有需要翻译的文本，直接返回原文
        if not text.strip():
            print("input text not exist valid content !")
            return text

        # 有效文本前后的空白字符提供，尽可能保留输入内容的格式
        prompt, start, end = self.build_prompt(background, document_format, target_language, text)
        result = self.llm(prompt)

        if result.content.strip() == "NOT_FOUNT_CONTENT":
            print("input text not found valid content !")
            return text

        return start + result.content + end

    def build_prompt(self, background, document_format, target_language, text):
        start, valid_text, end = self.extract_whitespace(text)
        system_message = self.build_system_message(background)
        user_message = self.build_user_message(document_format, target_language, valid_text)
        prompt = [system_message] + self.few_shot_example + [user_message]
        return prompt, start, end

    def predict_cost(self, document: TranscriptDocument, target_language=None, background=None, **kwargs):
        """
        预测翻译成本
        :param document:  待翻译的文档对象
        :param target_language: 目标语言
        :param background: 上下文背景知识
        :param kwargs: 其他关键参数
        :return:
        """
        # 使用 filter 过滤出 translate 属性为真的元素
        translate_piece = filter(lambda item: item.translate, document.get_pieces())

        # 使用 map 计算每个元素的 cost 并将结果转为列表
        predict_token_list = list(
            map(lambda item: self.calc_llm_token(background, item, target_language), translate_piece))
        # 汇总所有token
        #  {"prompt_token": prompt_token, "result_token": result_token, "length": len(translate_text)}
        # 初始化结果字典
        result = self.sum_document_token(predict_token_list)

        return result

    def sum_document_token(self, predict_token_list):
        result = {"prompt_token": 0, "result_token": 0, "length": 0}
        # 在一个循环中更新结果字典
        for item in predict_token_list:
            result["prompt_token"] += item["prompt_token"]
            result["result_token"] += item["result_token"]
            result["length"] += item["length"]
        return result

    def calc_llm_token(self, background, item, target_language):
        translate_text = self.build_translate_text(item.translate_content)
        # 需要注意translate_content 是list 可能是多行短文本,需要进行拼接
        prompt, _, _ = self.build_prompt(text=translate_text, background=background,
                                         document_format=item.format,
                                         target_language=target_language)
        # 获取提示词所需token
        prompt_token = self.llm.get_num_tokens_from_messages(prompt)
        # 获取回答预计所需token(与输入相当)
        result_token = self.llm.get_num_tokens(translate_text)
        return {"prompt_token": prompt_token, "result_token": result_token, "length": len(translate_text)}

    def build_translate_text(self, content_list: List[str]):
        # list 拼接为字符串
        pass
        return "\n" + self.separator + "\n".join(content_list)

    def build_user_message(self, document_format, target_language, text) -> HumanMessage:
        return HumanMessage(
            content=(self.build_message_content(text, target_language=target_language, content_format=document_format))
        )

    def build_system_message(self, background) -> SystemMessage:
        background = self.build_background(background)
        system_message = SystemMessage(content=(self.config['system'].format(background=background)))
        return system_message

    def build_background(self, background: str) -> str:
        if background is None or background.strip():
            return ""
        return self.config.get('background', '').format(background=background)

    def build_message_content(self, text, target_language=None, content_format=None):
        if target_language is None:
            target_language = self.default_language
        if content_format is None:
            content_format = self.default_format
        return f"L:{target_language}\nF:{content_format}\nC:{text}"

    def build_chat_sequence(self) -> [BaseMessage]:
        """

        :rtype: [BaseMessage]
        """
        sequence = []
        for example in self.config.get('example', []):
            sequence.append(
                HumanMessage(
                    content=self.build_message_content(example['user'], example['language'], example['format']))
            )
            sequence.append(AIMessage(content=example['llm']))
        return sequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translation Configuration")
    parser.add_argument('--config', default='./prompt.yaml', type=str,
                        help='Path to the configuration file.')
    args = parser.parse_args()

    chat = ChatOpenAI(model="gpt-3.5-turbo-16k")
    translator = DocumentTranslator(chat, args.config)
    res = translator.translate("用户输入的待翻译文本", target_language="English")
    print(res)
