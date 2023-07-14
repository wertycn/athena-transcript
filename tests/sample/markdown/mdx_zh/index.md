---
image: og/docs/quickstart-tutorial.jpg
sidebar_position: 0
title: Quickstart Tutorial
---

import Badges from '/_includes/badges.mdx';

<Badges/>

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import WCSoptionsWithAuth from '../../wcs/img/wcs-options-with-auth.png';
import WCScreateButton from '../../wcs/img/wcs-create-button.png';
import WCSApiKeyLocation from '../../wcs/img/wcs-apikey-location.png';

## 概述

欢迎。在这里，您将在大约20分钟内快速了解Weaviate。

您将：
- 构建一个向量数据库，以及
- 使用*语义搜索*进行查询。

:::info Object vectors
With Weaviate, you have options to:
- Have **Weaviate create vectors**, or
- Specify **custom vectors**.

This tutorial demonstrates both methods.
:::

#### 源数据

我们将使用一个（小型）测验数据集。

<details>
  <summary>我们使用的是哪些数据？</summary>

这些数据来自一个电视测验节目（"Jeopardy!"）。

|    | 分类       | 问题                                                                                                              | 答案                    |
|---:|:-----------|:------------------------------------------------------------------------------------------------------------------|:------------------------|
|  0 | 科学       | 这个器官从血液中去除多余的葡萄糖并储存为糖原                                                     | 肝脏                   |
|  1 | 动物       | 它是目前唯一一种属于长鼻目的活着的哺乳动物                                                      | 大象                  |
|  2 | 动物       | 相比鳄鱼，它的身体特征非常相似，唯独这一点例外                                                   | 鼻子或吻部             |
|  3 | 动物       | 体重约一吨，羚羊是非洲最大的这种动物物种                                                     | 羚羊                    |
|  4 | 动物       | 所有有毒蛇中最重的是这种北美响尾蛇                                                               | 钻石响尾蛇              |
|  5 | 科学       | 2000年新闻：Gunnison草原松鸡不仅仅是另一种北方草原松鸡，而是这个分类中的一种新物种             | 物种                    |
|  6 | 科学    | 一种"延展性"金属在冷却和压力下可以被拉成这个形状              |  导线                    |
|  7 | 科学    | 在1953年，沃森和克里克构建了这个物质的分子结构模型，这个物质携带基因                 |  DNA                     |
|  8 | 科学    | 对这个对流层的变化是导致我们天气变化的原因                |  大气层          |
|  9 | 科学    | 在70度的空气中，以大约每秒1130英尺的速度飞行的飞机打破了声音障碍                                      | 声音障碍           |

</details>

<hr/><br/>

## 创建一个实例

首先，创建一个Weaviate数据库。

1. 前往[WCS控制台](https://console.weaviate.cloud)，然后
    1. 点击<kbd>使用Weaviate云服务登录</kbd>。
    1. 如果您没有WCS帐户，请点击<kbd>注册</kbd>。
1. 使用您的WCS用户名和密码登录。
1. 点击 <kbd>创建集群</kbd>。

:::note <i class="fa-solid fa-camera-viewfinder"></i> <small>To create a WCS instance:</small>
<img src={WCScreateButton} width="100%" alt="Button to create WCS instance"/>
:::

<details>
  <summary>我可以使用其他方法吗？</summary>

可以。如果您喜欢其他方法，请查看我们的[安装选项](../installation/index.md)页面。

</details>


然后：

1. 选择<kbd>免费沙盒</kbd>层级。
2. 提供一个*集群名称*。
3. 将*启用身份验证？*设置为<kbd>是</kbd>。

:::note <i class="fa-solid fa-camera-viewfinder"></i> <small>Your selections should look like this:</small>
<img src={WCSoptionsWithAuth} width="100%" alt="Instance configuration"/>
:::

点击<kbd>创建</kbd>。这将需要大约2分钟的时间，完成后您将看到一个✔️的勾号。

#### 记下您的集群详情

您将需要：
- Weaviate URL
- 认证详情（Weaviate API密钥）

点击<kbd>详情</kbd>以查看它们。

对于Weaviate API密钥，点击<kbd><i class="fa-solid fa-key"></i></kbd>按钮。

:::note <i class="fa-solid fa-camera-viewfinder"></i> <small>Your WCS cluster details should look like this:</small>
<img src={WCSApiKeyLocation} width="60%" alt="Instance API key location"/>
:::

<hr/><br/>

## 安装客户端库

我们建议使用[Weaviate客户端库](../client-libraries/index.md)。要安装您首选的客户端库 <i class="fa-solid fa-down"></i>:

import CodeClientInstall from '/_includes/code/quickstart.clients.install.mdx';

:::info Install client libraries

<CodeClientInstall />

:::

<hr/><br/>

## 连接到Weaviate

从WCS的<kbd>详细信息</kbd>标签中获取以下内容：
- Weaviate的**API密钥**，以及
- Weaviate的**URL**。

由于我们将使用Hugging Face推理API生成向量，您还需要：
- 一个Hugging Face的**推理API密钥**。

因此，您可以按照以下方式实例化客户端：

import ConnectToWeaviateWithKey from '/_includes/code/quickstart.autoschema.connect.withkey.mdx'

<ConnectToWeaviateWithKey />

现在您已成功连接到Weaviate实例！

<hr/><br/>

## 定义一个类

接下来，我们定义一个数据集合（在Weaviate中称为“类”），用于存储对象：

import CodeAutoschemaMinimumSchema from '/_includes/code/quickstart.autoschema.minimum.schema.mdx'

<CodeAutoschemaMinimumSchema />

<details>
  <summary>如果我想使用不同的向量化模块怎么办？</summary>

在这个示例中，我们使用了`Hugging Face`的推理API。但是您也可以使用其他模块。

:::tip Our recommendation
Vectorizer selection is a big topic - so for now, we suggest sticking to the defaults and focus on learning the basics of Weaviate.
:::

如果您确实想要更改矢量化器，只需满足以下条件：
- 模块在您正在使用的Weaviate实例中可用，并且
- 您有该模块的API密钥（如果需要）。

以下每个模块都在免费沙盒中可用。

- `text2vec-cohere`
- `text2vec-huggingface`
- `text2vec-openai`
- `text2vec-palm`

根据您的选择，请确保通过设置适当的行将推理服务的 API 密钥传递给头部，记得用您的实际密钥替换占位符。

```js
"X-Cohere-Api-Key": "YOUR-COHERE-API-KEY",  // For Cohere
"X-HuggingFace-Api-Key": "YOUR-HUGGINGFACE-API-KEY",  // For Hugging Face
"X-OpenAI-Api-Key": "YOUR-OPENAI-API-KEY",  // For OpenAI
"X-Palm-Api-Key": "YOUR-PALM-API-KEY",  // For PaLM
```

另外，我们还提供了建议的`vectorizer`模块配置。

<Tabs groupId="inferenceAPIs">
<TabItem value="cohere" label="Cohere">

```json
class_obj = {
  "class": "Question",
  "vectorizer": "text2vec-cohere",
}
```

</TabItem>
<TabItem value="huggingface" label="Hugging Face">

```js
class_obj = {
  "class": "Question",
  "vectorizer": "text2vec-huggingface",
  "moduleConfig": {
    "text2vec-huggingface": {
      "model": "sentence-transformers/all-MiniLM-L6-v2",  // Can be any public or private Hugging Face model.
      "options": {
        "waitForModel": true,  // Try this if you get a "model not ready" error
      }
    }
  }
}
```

</TabItem>
<TabItem value="openai" label="OpenAI">

```js
class_obj = {
  "class": "Question",
  "vectorizer": "text2vec-openai",
  "moduleConfig": {
    "text2vec-openai": {
      "model": "ada",
      "modelVersion": "002",
      "type": "text"
    }
  }
}
```

</TabItem>
<TabItem value="palm" label="PaLM">

```js
class_obj = {
  "class": "Question",
  "vectorizer": "text2vec-palm",
  "moduleConfig": {
    "text2vec-palm": {
      "projectId": "YOUR-GOOGLE-CLOUD-PROJECT-ID",    // Required. Replace with your value: (e.g. "cloud-large-language-models")
      "apiEndpoint": "YOUR-API-ENDPOINT",             // Optional. Defaults to "us-central1-aiplatform.googleapis.com".
      "modelId": "YOUR-GOOGLE-CLOUD-MODEL-ID",        // Optional. Defaults to "textembedding-gecko".
    },
  }
}
```

</TabItem>
</Tabs>

</details>

这将创建一个名为`Question`的类，告诉Weaviate要使用哪个`vectorizer`，并为vectorizer设置`moduleConfig`。

:::tip Is a `vectorizer` setting mandatory?
- No. You always have the option of providing vector embeddings yourself.
- Setting a `vectorizer` gives Weaviate the option of creating vector embeddings for you.
    - If you do not wish to, you can set this to `none`.
:::

现在您已经准备好向Weaviate添加对象了。

<hr/><br/>

## 添加对象

我们将使用**批量导入**过程将对象添加到我们的Weaviate实例中。

<details>
  <summary>为什么使用批量导入？</summary>

批量导入提供了显著提高的导入性能，因此除非您有充分的理由不这样做（例如单个对象创建），否则几乎总是应该使用批量导入。

</details>

首先，您将使用`vectorizer`来创建对象向量。

### *选项1*：`vectorizer`

下面的代码在不使用向量的情况下传递对象数据。这会导致Weaviate使用指定的`vectorizer`为每个对象创建一个向量嵌入。

import CodeAutoschemaImport from '/_includes/code/quickstart.autoschema.import.mdx'

<CodeAutoschemaImport />

上面的代码：
- 加载对象，
- 初始化一个批处理过程，并
- 逐个将对象添加到目标类（`Question`）中。

### *选项2*：自定义`vector`

或者，您也可以为Weaviate提供自己的向量。

无论是否设置了`vectorizer`，如果指定了一个向量，Weaviate将使用它来表示对象。

import CodeAutoschemaImportCustomVectors from '/_includes/code/quickstart.autoschema.import.custom.vectors.mdx'

<CodeAutoschemaImportCustomVectors />

<details>
  <summary>使用<code>vectorizer</code>的自定义向量</summary>

请注意，您可以指定一个`vectorizer`，并且仍然可以提供一个自定义的向量。在这种情况下，请确保向量来自与`vectorizer`中指定的模型相同的模型。

在本教程中，它们来自于`sentence-transformers/all-MiniLM-L6-v2` - 与向量化器配置中指定的相同模型。

</details>

:::tip vector != object property
Do *not* specify object vectors as an object property. This will cause Weaviate to treat it as a regular property, rather than as a vector embedding.
:::

<hr/><br/>

# 将其组合起来

以下代码将上述步骤组合在一起。您可以运行它，将数据导入到您的Weaviate实例中。

<details>
  <summary>端到端代码</summary>

:::tip Remember to replace the **URL**, **Weaviate API key** and **inference API key**
:::

import CodeAutoschemaEndToEnd from '/_includes/code/quickstart.autoschema.endtoend.mdx'

<CodeAutoschemaEndToEnd />

恭喜，您已成功构建了一个向量数据库！

</details>

<hr/><br/>

## 查询

现在，我们可以运行查询。

### 语义搜索

让我们尝试进行相似性搜索。我们将使用`nearText`搜索来查找与`biology`最相似的测验对象。

import CodeAutoschemaNeartext from '/_includes/code/quickstart.autoschema.neartext.mdx'

<CodeAutoschemaNeartext />

您应该看到类似以下结果（这些结果可能会因使用的模块/模型而有所不同）：

import BiologyQuestionsJson from '/_includes/code/quickstart.biology.questions.mdx'

<BiologyQuestionsJson />

响应包含一个最相似于单词 `biology` 的向量的列表，包括前2个对象（由于设置了 `limit`）。

:::tip Why is this useful?
Notice that even though the word `biology` does not appear anywhere, Weaviate returns biology-related entries.

This example shows why vector searches are powerful. Vectorized data objects allow for searches based on degrees of similarity, as shown here.
:::

### 使用过滤器进行语义搜索

您可以在示例中添加一个布尔过滤器。例如，让我们运行相同的搜索，但只在具有"category"值为"ANIMALS"的对象中查找。

import CodeAutoschemaNeartextWithWhere from '/_includes/code/quickstart.autoschema.neartext.where.mdx'

<CodeAutoschemaNeartextWithWhere />

您应该会看到一个类似于以下内容的结果（这些结果可能因使用的模块/模型而异）：

import BiologyQuestionsWhereJson from '/_includes/code/quickstart.biology.where.questions.mdx'

<BiologyQuestionsWhereJson />

响应中包含了一个列表，其中包含与单词 `biology` 最相似的前两个对象（由于设置了 `limit`）- 但仅限于 "ANIMALS" 类别。

:::tip Why is this useful?
Using a Boolean filter allows you to combine the flexibility of vector search with the precision of `where` filters.
:::


<!-- 注意：添加了生成式搜索示例；但现在将其隐藏起来，因为它会使新用户的工作流程变得非常困难。1）他们现在需要一个OpenAI/Cohere密钥。2）模式需要包括一个生成式模块的定义。3）生成式API的速率限制很低，可能会很麻烦。 -->

<!-- ### 生成式搜索

现在让我们尝试一下生成式搜索。我们将像上面那样检索一组结果，然后使用一个LLM以简明的方式解释每个答案。

import CodeAutoschemaGenerative from '/_includes/code/quickstart.autoschema.generativesearch.mdx'

<CodeAutoschemaGenerative />

您应该看到类似于以下结果（根据使用的模型可能会有所不同）：

import BiologyGenerativeSearchJson from '/_includes/code/quickstart.biology.generativesearch.mdx'

<BiologyGenerativeSearchJson />

在这里，我们可以看到Weaviate检索到了与之前相同的结果。但现在它还包括了一个额外的生成文本，其中包含了每个答案的以简明易懂的语言解释。

:::tip Why is this useful?
Generative search sends retrieved data from Weaviate to a large language model (LLM). This allows you to go beyond simple data retrieval, but transform the data into a more useful form, without ever leaving Weaviate.
::: -->

<hr/><br/>

## 总结

干得好！你已经做到了：
- 使用Weaviate创建了自己的基于云的向量数据库，
- 用数据对象填充它，
    - 使用推理API，或者
    - 使用自定义向量，
- 执行了文本相似性搜索。

接下来的路在你手里。我们在下面提供了一些链接，或者你可以查看侧边栏。

<!-- TODO - 提供一些具体的“中级”学习路径 -->

## 故障排除和常见问题

我们在下面提供了一些常见问题或潜在问题的答案。

#### 如何确认类的创建

<details>
  <summary>查看答案</summary>

如果您不确定类是否已创建，您可以通过访问[`schema`端点](../api/rest/schema.md)来确认（将URL替换为实际的端点）：

```
https://some-endpoint.weaviate.network/v1/schema
```

您应该看到：

```json
{
    "classes": [
        {
            "class": "Question",
            ...  // truncated additional information here
            "vectorizer": "text2vec-huggingface"
        }
    ]
}
```

模式应该表明已添加`Question`类。

:::note REST & GraphQL in Weaviate
Weaviate uses a combination of RESTful and GraphQL APIs. In Weaviate, RESTful API endpoints can be used to add data or obtain information about the Weaviate instance, and the GraphQL interface to retrieve data.
:::

</details>

#### 如果您看到 <code>Error: Name 'Question' already used as a name for an Object class</code>

<details>
  <summary>查看答案</summary>

如果您尝试创建一个在您的Weaviate实例中已经存在的类别，您可能会看到这个错误。在这种情况下，您可以按照下面的说明删除该类别。

import CautionSchemaDeleteClass from '/_includes/schema-delete-class.mdx'

<CautionSchemaDeleteClass />

</details>

#### 如何确认数据导入

<details>
  <summary>查看答案</summary>

要确认数据成功导入，请导航到[`objects`端点](../api/rest/objects.md)检查所有对象是否已导入（用您实际的端点进行替换）。

```
https://some-endpoint.weaviate.network/v1/objects
```

您应该看到：

```json
{
    "deprecations": null,
    "objects": [
        ...  // Details of each object
    ],
    "totalResults": 10  // You should see 10 results here
}
```

您应该能够确认您已导入了所有的`10`个对象。

</details>

#### 如果`nearText`搜索不起作用

<details>
  <summary>查看答案</summary>

要执行基于文本的(`nearText`)相似性搜索，您需要启用一个向量化器，并在您的类中进行配置。

请确保您按照[此部分](#define-a-class)所示进行了配置。

如果仍然无法正常工作，请[联系我们](#more-resources)！

</details>

#### 我的沙盒会被删除吗？

<details>
  <summary>注意：沙盒到期和选项</summary>

import SandBoxExpiry from '/_includes/sandbox.expiry.mdx';

<SandBoxExpiry/>

</details>

## 下一步

import WhatNext from '/_includes/quickstart.what-next.mdx';

<WhatNext />

## 更多资源

import DocsMoreResources from '/_includes/more-resources-docs.md';

<DocsMoreResources />