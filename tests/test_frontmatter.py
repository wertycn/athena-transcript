import unittest
import frontmatter
import mistune


class MyTestCase(unittest.TestCase):
    def test_something(self):
        with open('sample/markdown/frontmatter/yaml-long-content.md') as f:
            post = frontmatter.loads(f.read())
            print(post)

    def test_mistune(self):
        with open('sample/markdown/frontmatter/yaml-long-content.md') as f:
            post = frontmatter.loads(f.read())

        markdown = mistune.markdown(post.content)
        print(markdown)


if __name__ == '__main__':
    unittest.main()
