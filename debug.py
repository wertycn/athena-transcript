
from pathlib import Path
from langchain_openai import ChatOpenAI

from athena_transcript.AthenaTranscript import AthenaTranscript
from athena_transcript.DocumentTranslator import DocumentTranslator


if __name__ == '__main__':
   
    chat = ChatOpenAI(model="glm-4")
    translator = DocumentTranslator(chat, "/workspaces/athena-transcript/prompt.yaml")
    # res = translator.translate("""Type "help", "copyright", "credits" or "license" for more information.""", target_language="中文")
    # print(res)


    transcript = AthenaTranscript(
        translator=translator,
        source_path=Path("/workspaces/vueuse"),
        target_path=Path("/workspaces/result_zh/vueuse"),
        excludes="*-cn.md,*_cn.md")
    transcript.translate()