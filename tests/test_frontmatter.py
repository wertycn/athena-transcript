import unittest
import frontmatter

class MyTestCase(unittest.TestCase):
    def test_something(self):
        with open('sample/markdown/frontmatter/yaml-long-content.md') as f:
            post = frontmatter.loads(f.read())
            print(post)


if __name__ == '__main__':
    unittest.main()
