import unittest

from parser.java_parser import JavaParser
from parser.js_parser import JavaScriptParser
from parser.python_parser import PythonParser


class TestJavaParser(unittest.TestCase):
    def test_method_line_numbers_are_based_on_file_content(self):
        source = "\n".join(
            [
                "package demo;",
                "",
                "public class AuthService {",
                "    public boolean login(String username) {",
                "        if (username == null) {",
                "            return false;",
                "        }",
                "        return true;",
                "    }",
                "",
                "    private static String token() {",
                "        return \"jwt\";",
                "    }",
                "}",
            ]
        )

        parser = JavaParser("demo-repo")
        chunks = parser.parse_file("AuthService.java", source, "src/AuthService.java")
        methods = {chunk.function_name: chunk for chunk in chunks if chunk.chunk_type == "method"}

        self.assertEqual(methods["login"].start_line, 4)
        self.assertEqual(methods["login"].end_line, 9)
        self.assertEqual(methods["token"].start_line, 11)
        self.assertEqual(methods["token"].end_line, 13)
        self.assertEqual(methods["login"].content, "\n".join(source.splitlines()[3:9]))


class TestJavaScriptParser(unittest.TestCase):
    def test_class_method_extraction_uses_full_file_content(self):
        source = "\n".join(
            [
                "export class AuthService {",
                "    constructor(client) {",
                "        this.client = client;",
                "    }",
                "",
                "    async login(username) {",
                "        if (!username) {",
                "            return null;",
                "        }",
                "        return this.client.login(username);",
                "    }",
                "",
                "    validateJwt(token) {",
                "        return token.length > 0;",
                "    }",
                "}",
            ]
        )

        parser = JavaScriptParser("demo-repo")
        chunks = parser.parse_file("AuthService.js", source, "src/AuthService.js")
        methods = {chunk.function_name: chunk for chunk in chunks if chunk.chunk_type == "method"}

        self.assertEqual(methods["constructor"].start_line, 2)
        self.assertEqual(methods["constructor"].end_line, 4)
        self.assertEqual(methods["login"].start_line, 6)
        self.assertEqual(methods["login"].end_line, 11)
        self.assertEqual(methods["validateJwt"].start_line, 13)
        self.assertEqual(methods["validateJwt"].end_line, 15)
        self.assertIn("return this.client.login(username);", methods["login"].content)


class TestPythonParser(unittest.TestCase):
    def test_methods_with_same_name_in_different_classes_get_unique_ids(self):
        source = "\n".join(
            [
                "class First:",
                "    def __init__(self):",
                "        self.name = 'first'",
                "",
                "class Second:",
                "    def __init__(self):",
                "        self.name = 'second'",
            ]
        )

        parser = PythonParser("demo-repo")
        chunks = parser.parse_file("dup_ids.py", source, "src/dup_ids.py")
        init_methods = [chunk for chunk in chunks if chunk.function_name == "__init__"]

        self.assertEqual(len(init_methods), 2)
        self.assertNotEqual(init_methods[0].chunk_id, init_methods[1].chunk_id)


if __name__ == "__main__":
    unittest.main()
