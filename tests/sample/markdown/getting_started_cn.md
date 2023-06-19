# 快速上手指南


本教程将为您快速介绍如何使用LangChain构建一个端到端的语言模型应用程序。

## 安装

要开始使用LangChain，请使用以下命令进行安装：

```bash
pip install langchain
# or
conda install langchain -c conda-forge
```


## 环境设置

使用LangChain通常需要与一个或多个模型提供商、数据存储、API等进行集成。

在这个示例中，我们将使用OpenAI的API，因此首先需要安装他们的SDK：

```bash
pip install openai
```

然后我们需要在终端中设置环境变量。

```bash
export OPENAI_API_KEY="..."
```

或者，您可以在Jupyter笔记本（或Python脚本）中执行此操作：

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```


## 构建语言模型应用程序：LLMs

现在我们已经安装了LangChain并设置好了环境，我们可以开始构建我们的语言模型应用程序了。

LangChain提供了许多模块，可以用来构建语言模型应用程序。这些模块可以组合在一起创建更复杂的应用程序，也可以单独用于简单的应用程序。

## LLMs：从语言模型获取预测结果

LangChain的最基本的构建单元是在一些输入上调用LLM。让我们通过一个简单的例子来演示如何做到这一点。
为此，让我们假设我们正在构建一个根据公司的产品生成公司名称的服务。

为了做到这一点，我们首先需要导入LLM包装器。

```python
from langchain.llms import OpenAI
```

然后，我们可以使用任何参数来初始化包装器。
在这个例子中，我们可能希望输出更加随机，所以我们将使用较高的温度进行初始化。

```python
llm = OpenAI(temperature=0.9)
```

现在我们可以对一些输入进行调用了！

```python
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
```

```pycon
Feetful of Fun
```

有关如何在LangChain中使用LLMs的更多详细信息，请参阅[LLM入门指南](../modules/models/llms/getting_started.ipynb)。

## 提示模板：管理LLM的提示

调用LLM是一个很好的第一步，但这只是个开始。
通常，在应用程序中使用LLM时，您不会直接将用户输入发送到LLM。
相反，您可能会将用户输入与提示一起构建，然后将其发送给LLM。

例如，在之前的示例中，我们传递的文本是硬编码的，用于要求一个制造多彩袜子的公司的名称。
在这个想象的服务中，我们希望只接收用户输入来描述公司的事务，然后使用该信息格式化提示。

使用LangChain实现这一点非常容易！

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


[了解更多细节，请查看提示的入门指南。](../modules/prompts/chat_prompt_template.ipynb)

## 链：在多步骤工作流中组合LLM和提示

到目前为止，我们已经单独使用了PromptTemplate和LLM这两个基本元素。但是，一个真实的应用程序不仅仅是一个基本元素，而是它们的组合。

在LangChain中，链由链接组成，可以是LLM或其他链等基元。

最核心的链类型是LLMChain，它由PromptTemplate和LLM组成。

扩展之前的示例，我们可以构建一个LLMChain，它接受用户输入，使用PromptTemplate对其进行格式化，然后将格式化的响应传递给LLM。

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

我们现在可以创建一个非常简单的链条，它将接收用户输入，将输入格式化为提示信息，然后将其发送到LLM：

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

现在我们可以仅指定产品来运行该链！

```python
chain.run("colorful socks")
# -> '\n\nSocktastic!'
```

这就是第一个链条 - 一个LLM链条。
这是较简单类型的链条之一，但是理解它的工作原理将为您处理更复杂的链条奠定良好基础。

[要获取更多详细信息，请查看链条入门指南。](../modules/chains/getting_started.ipynb)

## Agents：根据用户输入动态调用链条

到目前为止，我们看过的链条都是按照预定顺序运行的。

而代理不再如此：它们使用LLM来确定采取的动作及其顺序。动作可以是使用工具并观察其输出，或者返回给用户。

正确使用代理可以非常强大。在本教程中，我们将向您展示如何通过最简单、最高级别的API轻松使用代理。


为了加载代理，您应该了解以下概念：

- 工具（Tool）：执行特定任务的函数。例如：Google搜索、数据库查找、Python REPL、其他链。目前，工具的接口是期望一个字符串作为输入，并返回一个字符串作为输出的函数。
- LLM：为代理提供动力的语言模型。
- 代理（Agent）：要使用的代理。这应该是一个引用支持代理类的字符串。因为这个笔记本重点介绍最简单、最高级的API，所以只涵盖了使用标准支持的代理。如果您想实现自定义代理，请参阅自定义代理的文档（即将推出）。

**代理**：有关支持的代理及其规格的列表，请参见[这里](../modules/agents/agents.md)。

**工具**：有关预定义工具及其规格的列表，请参见[这里](../modules/agents/tools.md)。

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

到目前为止，我们所介绍的所有链和代理都是无状态的。但是，通常情况下，您可能希望链或代理具有某种"记忆"的概念，以便它可以记住有关其先前交互的信息。最明显和简单的例子就是设计一个聊天机器人 - 您希望它记住先前的消息，以便在对话中使用上下文来进行更好的交流。这将是一种"短期记忆"的形式。在更复杂的情况下，您可以想象一个链/代理随着时间的推移记住关键信息 - 这将是一种"长期记忆"的形式。有关后者的更具体的想法，请参见这篇[精彩论文](https://memprompt.com/)。

LangChain提供了几种专门为此目的创建的链。本文档演示了如何使用其中一种链（`ConversationChain`）以及两种不同类型的记忆。

默认情况下，`ConversationChain`具有一种简单类型的内存，它会记住所有先前的输入/输出并将它们添加到传递的上下文中。让我们来看看如何使用这个链（设置`verbose=True`以便我们可以看到提示）。

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

## 构建语言模型应用程序：聊天模型

类似地，您可以使用聊天模型而不是LLM（语言模型）。聊天模型是语言模型的一种变体。虽然聊天模型在底层使用语言模型，但它们提供的接口有所不同：它们不仅提供了“输入文本，输出文本”的API，还提供了一种以“聊天消息”为输入和输出的接口。

聊天模型的API相对较新，因此我们仍在探索正确的抽象层次。

## 从聊天模型获取消息完成

您可以通过将一个或多个消息传递给聊天模型来获取聊天完成。响应将是一条消息。LangChain目前支持的消息类型有`AIMessage`、`HumanMessage`、`SystemMessage`和`ChatMessage`，其中`ChatMessage`接受一个任意的角色参数。大多数情况下，您只需要处理`HumanMessage`、`AIMessage`和`SystemMessage`。

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

您还可以向OpenAI的gpt-3.5-turbo和gpt-4模型传递多个消息。

```python
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

您可以进一步进行操作，使用`generate`为多组消息生成补全。这将返回一个带有额外`message`参数的`LLMResult`。
```python
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

您可以从这个LLMResult中恢复诸如令牌使用情况等内容。
```
result.llm_output['token_usage']
# -> {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}
```


## 聊天提示模板
与LLMs类似，您可以使用`MessagePromptTemplate`来使用模板。您可以从一个或多个`MessagePromptTemplate`构建一个`ChatPromptTemplate`。您可以使用`ChatPromptTemplate`的`format_prompt`方法，它返回一个`PromptValue`，您可以将其转换为字符串或`Message`对象，具体取决于您是否希望将格式化的值作为输入传递给LLM或聊天模型。

为了方便起见，模板上暴露了一个`from_template`方法。如果您要使用此模板，它将如下所示：

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

## 使用聊天模型的链路

上一节讨论的`LLMChain`也可以与聊天模型一起使用：

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

## 使用聊天模型的代理人
代理人也可以与聊天模型一起使用，您可以使用`AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION`作为代理人类型进行初始化。

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
## 内存：为链和代理添加状态

您可以在使用聊天模型初始化的链和代理中使用内存。与用于LM的内存相比，主要区别在于我们可以将其保留为独立的内存对象，而不是将所有先前的消息压缩成一个字符串。

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

