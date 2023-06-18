# 快速上手指南

本教程将为您快速介绍如何使用LangChain构建一个端到端的语言模型应用程序。
## 安装
要开始使用，请使用以下命令安装LangChain：
```bash
pip install langchain
# or
conda install langchain -c conda-forge
```


## 环境设置
使用LangChain通常需要与一个或多个模型提供商、数据存储、API等进行集成。
对于这个示例，我们将使用OpenAI的API，因此我们首先需要安装他们的SDK：
```bash
pip install openai
```

然后，我们需要在终端中设置环境变量。
```bash
export OPENAI_API_KEY="..."
```

或者，您也可以在Jupyter笔记本（或Python脚本）中执行以下操作：
```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```


## 构建语言模型应用程序：LLMs
现在我们已经安装了LangChain并设置好了环境，我们可以开始构建我们的语言模型应用程序了。
LangChain提供了许多模块，可以用来构建语言模型应用程序。这些模块可以组合在一起创建更复杂的应用程序，也可以单独用于简单的应用程序。


## LLMs: 从语言模型获取预测结果
LangChain的最基本组成部分是对一些输入调用LLM（语言模型）。让我们通过一个简单的例子来演示如何做到这一点。为此，让我们假设我们正在构建一个基于公司所生产的产品来生成公司名称的服务。
为了做到这一点，我们首先需要导入LLM包装器。
```python
from langchain.llms import OpenAI
```

然后，我们可以使用任何参数来初始化包装器。在这个例子中，我们可能希望输出更加随机，所以我们将使用较高的温度进行初始化。
```python
llm = OpenAI(temperature=0.9)
```

我们现在可以对一些输入进行调用了！
```python
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
```

```pycon
Feetful of Fun
```

有关如何在LangChain中使用LLM的更多详细信息，请参阅[LLM入门指南](../modules/models/llms/getting_started.ipynb)。

## 提示模板: 管理LLM的提示
调用LLM是一个很好的第一步，但这只是个开始。通常情况下，在应用程序中使用LLM时，您不会直接将用户输入发送到LLM。相反，您可能正在接收用户输入并构建提示，然后将其发送到LLM。
例如，在前面的示例中，我们传递的文本是硬编码的，用于询问一个制造彩色袜子的公司的名称。在这个虚构的服务中，我们只需要获取用户输入的公司描述信息，并将其格式化后作为提示信息。
使用LangChain非常容易实现这个目标！
首先，让我们定义提示模板：
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

现在让我们看看这是如何工作的！我们可以调用`.format`方法来进行格式化。
```python
print(prompt.format(product="colorful socks"))
```

```pycon
What is a good name for a company that makes colorful socks?
```


[查看更多详细信息，请参阅提示的入门指南。](../modules/prompts/chat_prompt_template.ipynb)



## 链: 在多步骤工作流中组合LLMs和提示信息
到目前为止，我们只是单独使用了PromptTemplate和LLM两个原语。但是，一个真正的应用程序当然不只是一个原语，而是它们的组合。
LangChain中的链由链接组成，这些链接可以是像LLMs这样的原始链或其他链。
最核心的链式类型是LLMChain，它由PromptTemplate和LLM组成。
在扩展之前的示例的基础上，我们可以构建一个LLMChain，它接受用户输入，使用PromptTemplate对其进行格式化，然后将格式化后的响应传递给LLM。
```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

我们现在可以创建一个非常简单的链条，它将接收用户输入，将其格式化为提示信息，并将其发送给LLM。
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

现在我们可以只指定产品运行该链！
```python
chain.run("colorful socks")
# -> '\n\nSocktastic!'
```

这就是了！这是第一个链 - 一个LLM链。这是较简单类型的链之一，但是理解它的工作原理将为您更好地处理更复杂的链提供基础。
[更多详细信息，请查看链的入门指南。](../modules/chains/getting_started.ipynb)
## 代理人: 根据用户输入动态调用链


到目前为止，我们看到的链式结构是按照预定的顺序运行的。
代理程序不再执行某些操作：它们使用LLM来确定采取哪些行动以及以何种顺序进行。行动可以是使用工具并观察其输出，也可以是返回给用户。
当正确使用时，代理可以非常强大。在本教程中，我们将向您展示如何通过最简单、最高级别的API轻松使用代理。

为了加载代理程序，您应该了解以下概念：
- 工具：执行特定任务的函数。这可以是诸如：Google搜索、数据库查找、Python REPL、其他链等。工具的接口目前是一个期望输入为字符串，输出为字符串的函数。- LLM: 为该代理程序提供动力的语言模型。- Agent: 要使用的代理。这应该是一个字符串，引用一个支持代理类。因为这个笔记本着重介绍最简单、最高级别的API，所以只涵盖了使用标准支持的代理。如果您想实现一个自定义的代理，请参阅自定义代理的文档（即将推出）。
**Agents**：有关支持的代理和其规格的列表，请参阅[这里](../modules/agents/agents.md)。
**工具**：有关预定义工具及其规格的列表，请参见[此处](../modules/agents/tools.md)。
对于这个示例，您还需要安装SerpAPI Python包。
```bash
pip install google-search-results
```

并设置适当的环境变量。
```python
import os
os.environ["SERPAPI_API_KEY"] = "..."
```

现在我们可以开始了！
```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
```

```pycon
> Entering new AgentExecutor chain...
 I need to find the temperature first, then use the calculator to raise it to the .023 power.
Action: Search
Action Input: "High temperature in SF yesterday"
Observation: San Francisco Temperature Yesterday. Maximum temperature yesterday: 57 °F (at 1:56 pm) Minimum temperature yesterday: 49 °F (at 1:56 am) Average temperature ...
Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
Action: Calculator
Action Input: 57^.023
Observation: Answer: 1.0974509573251117

Thought: I now know the final answer
Final Answer: The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

> Finished chain.
```



## 内存：为链和代理添加状态
到目前为止，我们所涉及的所有链条和代理都是无状态的。但是，通常情况下，您可能希望链条或代理具有一些"记忆"的概念，以便它可以记住有关其之前交互的信息。最清晰和简单的例子是设计聊天机器人 - 您希望它记住以前的消息，以便在对话中使用上下文进行更好的交流。这将是一种"短期记忆"。在更复杂的一面，您可以想象一种链条/代理随时间记住关键信息 - 这将是一种"长期记忆"的形式。有关后者的更具体的想法，请参阅这个[精彩的论文](https://memprompt.com/)。
LangChain为此目的提供了几个专门创建的链。本教程将演示如何使用其中一个链（`ConversationChain`）并使用两种不同类型的记忆。
默认情况下，`ConversationChain`拥有一个简单类型的记忆，它会记住所有之前的输入/输出，并将它们添加到传递的上下文中。让我们来看看如何使用这个链式结构（设置`verbose=True`以便我们可以看到提示信息）。
```python
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished chain.
' Hello! How are you today?'
```

```python
output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:  Hello! How are you today?
Human: I'm doing well! Just having a conversation with an AI.
AI:

> Finished chain.
" That's great! What would you like to talk about?"
```

## 构建一个语言模型应用程序：聊天模型
类似地，您可以使用聊天模型而不是语言模型。聊天模型是语言模型的一种变体。虽然聊天模型在内部使用语言模型，但它们提供的接口略有不同：它们不是以“输入文本，输出文本”的方式暴露API，而是以“聊天消息”作为输入和输出的接口。
聊天模型的API还比较新，因此我们仍在探索正确的抽象。
## 从聊天模型中获取消息补全
通过向聊天模型传递一个或多个消息，您可以获得聊天完成。响应将是一条消息。目前在LangChain中支持的消息类型有`AIMessage`、`HumanMessage`、`SystemMessage`和`ChatMessage` -- `ChatMessage`接受一个任意的角色参数。大多数情况下，您只需要处理`HumanMessage`、`AIMessage`和`SystemMessage`。
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
```

您可以通过传入单个消息来获取补全结果。
```python
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

您还可以为OpenAI的gpt-3.5-turbo和gpt-4模型传入多个消息。
```python
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

您可以进一步使用`generate`生成多组消息的完成。这将返回一个带有额外`message`参数的`LLMResult`对象：```python
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result
# -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}})
```

您可以从此LLMResult中恢复诸如令牌使用情况之类的内容。```
result.llm_output['token_usage']
# -> {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}
```


## 聊天提示模板与LLM类似，您可以使用`MessagePromptTemplate`来利用模板。您可以从一个或多个`MessagePromptTemplate`构建一个`ChatPromptTemplate`。您可以使用`ChatPromptTemplate`的`format_prompt`方法，它会返回一个`PromptValue`，您可以将其转换为字符串或`Message`对象，取决于您是否希望将格式化的值用作LLM或Chat模型的输入。
为了方便起见，模板上暴露了一个`from_template`方法。如果您要使用这个模板，它将如下所示：
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

## 聊天模型中的链式结构上述部分讨论的 `LLMChain` 也可以与聊天模型一起使用：
```python
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="French", text="I love programming.")
# -> "J'aime programmer."
```

## 带有聊天模型的代理代理也可以与聊天模型一起使用，您可以使用`AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION`作为代理类型进行初始化。
```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
chat = ChatOpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
```

```pycon

> Entering new AgentExecutor chain...
Thought: I need to use a search engine to find Olivia Wilde's boyfriend and a calculator to raise his age to the 0.23 power.
Action:
{
  "action": "Search",
  "action_input": "Olivia Wilde boyfriend"
}

Observation: Sudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.
Thought:I need to use a search engine to find Harry Styles' current age.
Action:
{
  "action": "Search",
  "action_input": "Harry Styles age"
}

Observation: 29 years
Thought:Now I need to calculate 29 raised to the 0.23 power.
Action:
{
  "action": "Calculator",
  "action_input": "29^0.23"
}

Observation: Answer: 2.169459462491557

Thought:I now know the final answer.
Final Answer: 2.169459462491557

> Finished chain.
'2.169459462491557'
```
## 内存：为链和代理添加状态您可以使用已初始化了聊天模型的链和代理与内存一起使用。与用于语言模型的内存不同的是，我们可以将先前的所有消息保留为独立的内存对象，而不是尝试将它们压缩为一个字符串。
```python
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

conversation.predict(input="Hi there!")
# -> 'Hello! How can I assist you today?'


conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
# -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

conversation.predict(input="Tell me about yourself.")
# -> "Sure! I am an AI language model created by OpenAI. I was trained on a large dataset of text from the internet, which allows me to understand and generate human-like language. I can answer questions, provide information, and even have conversations like this one. Is there anything else you'd like to know about me?"
```

