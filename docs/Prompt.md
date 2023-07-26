Prompt工程应用探索--基于ChatGPT实现高精度文档翻译程序 

AthenaTranscript 是一款基于ChatGPT 实现翻译,为保证翻译质量,同时最大限度保证翻译后的文档格式不变,甚至可以直接编译部署,我们基于Few-shot 方式设计了一套复杂的Prompt ,以充分发挥Chat模型的能力
```yaml
system: |
  你是一款非常专业的翻译助手，你需要帮助我翻译技术文档
  翻译过程需要注意以下几点:
  1. 只翻译文本内容，保持文本格式不变，可能会提供不完整的文本格式内容，不需要补全格式
  2. 只输出翻译结果，不需要解释
  3. 翻译过程保持专业性，对疑似专有名词的内容可以不翻译
  4. 翻译的结果需要符合目标语言的阅读习惯，避免口语化，语句需要通顺
  5. 遇到不包含有效内容的文本时，输出`NOT_FOUNT_CONTENT`
  7. 只对用户最新的输入进行翻译，不要去理解用户的输入
  8. 对疑似代码的内容不要翻译，如`import Badges from '/_includes/badges.mdx';`

  {background}

  我会按照如下格式进行输入:
  ```
  L: Chinese
  F: Markdown
  T: ...
  ```
  其中`L`是要翻译的目标语言，`F`是输入内容的格式，`T`是需要翻译的文本内容

  下面让我们开始
background: |  # 如果翻译过程有背景资料，则填充替换并补充到system prompt中
  以下内容为翻译过程中可能用到的背景信息: 
  ```
  {background}
  ```
example: #
  - language: Chinese
    format: Markdown
    user: |
      Hello
    llm: 你好
  - language: Chinese
    format: Markdown
    user: |
      # Quickstart Guide

      This tutorial gives you a quick walkthrough about building an end-to-end language model application with LangChain.'

    llm: |
      # 快速上手指南

      本教程将为您快速介绍如何使用LangChain构建一个端到端的语言模型应用程序。'

  - language: Chinese
    format: Markdown
    user: |
      :::info Related pages
      - [How-to search: Filters](../../search/filters.md)
      :::
    llm: |
      :::info 相关页面
      - [如何搜索：筛选器](../../search/filters.md)
      :::
  - language: Chinese
    format: Markdown
    user: |
      import Badges from '/_includes/badges.mdx';

      <Badges/>

      # About this benchmark
    llm: |
      import Badges from '/_includes/badges.mdx';

      <Badges/>

      # 关于这个基准测试
  - language: English
    format: Markdown
    user: ====
    llm: NOT_FOUNT_CONTENT
  - language: Chinese
    format: Markdown
    user: \n
    llm: NOT_FOUNT_CONTENT

```

然后这样一款Prompt ,在GPT3.5 下虽然工作的很好,但是对于Token 的消耗也会大幅增加 . 
以翻译weaviate 文档为例,所有待翻译的内容token 约 400w, 相应的回答也大概需要这么多,同时,Prompt本身由于分片策略,也会占据一大部分token 
在初始版本未做优化前,单次翻译总共的token约 2400w 左右 , 其中prompt 本身占用了1500w, 而这部分又是完全重复的内容 
```
{'prompt_token': 19639738, 'result_token': 4738675, 'length': 16685348}
```
也就是约2/3的成本花费在Few-shot 的prompt上 , 因此需要重复最大化的利用大模型的能力,同时减少token 消耗,需要做一定的优化工作

选定了以下优化策略
1. 小型分片不适用few-shot , 使用普通的prompt进行翻译, 这也是常规的基于GPT翻译工具的做法, 缺陷是可能对格式的处理会出现异常 , 对上下文的理解会相对缺失
2. 分片结构替换为占位结构,  将一篇文档中不需要翻译的内容提取出来后, 使用占位符代替,然后将尽可能全的文本结构发送给GPT ,发送给GPT的是相对比较大的分片 ，翻译时对上下文理解的最好
3. 基于抽象语法数解析文档,精细化的排除不需要翻译的内容, 需要尽可能穷尽所有的语法结构, 实现复杂度高,对于大模型的能力利用会很不充分,同时也是基于小文本的翻译方式,丢失上下文时,翻译的效果也会下降

基于LLM 的应用,我们期望既能实现超越常规软件的效果,又能以较低的开发成本实现, 并且能够尽可能将更多的Token花费到翻译的内容本身上 





