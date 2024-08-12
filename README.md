# AthenaTranscript

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/your_username/project_name.svg?branch=master)](https://travis-ci.org/your_username/project_name)
[![Coverage Status](https://coveralls.io/repos/github/your_username/project_name/badge.svg?branch=master)](https://coveralls.io/github/your_username/project_name?branch=master)

AthenaTranscript 是一款基于LLM技术的文档翻译程序，专为软件开发者设计。它以希腊神话中的智慧女神雅典娜命名，象征着智慧和专业性。

帮助开发者准确、高效地翻译技术文档。通过深入理解文档的语义和上下文，它将技术文档转化为准确、流畅的翻译结果。

## 功能特性

- [x] 基于OpenAI ChatGPT 作为底层翻译工具，翻译结果较传统机翻效果更好
- [x] 支持Markdown, NoteBook , RST 等多种格式技术文档翻译，翻译过程保持文件格式不变
- [ ] 支持Mdx结构化解析
- [ ] 支持指定git项目，记录翻译进度，再项目文档更新后进行变更翻译，后续还需支持文件内部的增量翻译 -> 基于分片解决
- [ ] 断点续翻，异常中断后，从上次翻译结果开始， 不需要重新更新
- [ ] 提供Github Action 支持
- [ ] 提供Web服务支持

## 优化计划

- [ ] 支持Token 预计算及执行完成统计，用于评估单次翻译任务的成本
- [ ] 支持多个小分片合并后一起翻译，避免多个被代码块分隔后多个小分片重复请求导致得大量token被用于prompt而不是带翻译文档本身
- [ ] 单行分片使用普通prompt 进行翻译 以减少token使用 
- [ ] 记录文档分片和翻译结果分片， 建立文档行位置索引，用于支持增量翻译及局部翻译优化
- [ ] 基于抽象语法树提供更精准的翻译结构处理
- [ ] 需要独立路径记录文档的翻译记录
- [ ] frontmatter 头翻译


## 安装

TODO: 提供项目的安装指南和依赖项。

### 依赖项

TODO: 列出项目所需的依赖项。

### 安装步骤

```bash
pip install athena-transript
```

## 快速开始

TODO :

```python
# import AthenaTranscript


# 使用示例
```

## 示例

提供一些使用示例或代码片段，演示项目的不同功能和用法。

```python
# 代码示例
```

## 文档

链接到项目的详细文档或API文档。

## 贡献

说明如何贡献到项目或参与其中。

## 版权和许可证

提供项目的版权信息和许可证类型。

## 作者

列出项目的作者或维护者。

## 感谢

致谢和鸣谢项目中对你有帮助的人、组织或项目。

## 常见问题

列出一些常见问题和解答，以帮助用户解决问题。

## 变更日志

列出项目的版本历史和变更日志。

## 社区和联系方式

提供社区和联系方式，如论坛、邮件列表或Slack频道。

## 支持项目

如果适用，提供支持项目的捐赠方式，如Patreon、OpenCollective等。
