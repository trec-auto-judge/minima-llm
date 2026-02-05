"""Tests for minima_llm.dspy_adapter module, particularly TolerantChatAdapter."""

import pytest
from typing import List, Literal, Optional, Union
from textwrap import dedent

# Skip all tests in this module if dspy is not installed
pytest.importorskip("dspy")


class TestIsListStr:
    """Tests for TolerantChatAdapter.is_list_str() type detection."""

    @pytest.fixture
    def adapter_class(self):
        """Import TolerantChatAdapter (requires dspy)."""
        from minima_llm.dspy_adapter import TolerantChatAdapter
        return TolerantChatAdapter

    def test_list_str_lowercase(self, adapter_class):
        """list[str] should be detected."""
        assert adapter_class.is_list_str(list[str]) is True

    def test_list_str_typing(self, adapter_class):
        """List[str] from typing should be detected."""
        assert adapter_class.is_list_str(List[str]) is True

    def test_optional_list_str_lowercase(self, adapter_class):
        """Optional[list[str]] should be detected."""
        assert adapter_class.is_list_str(Optional[list[str]]) is True

    def test_optional_list_str_typing(self, adapter_class):
        """Optional[List[str]] should be detected."""
        assert adapter_class.is_list_str(Optional[List[str]]) is True

    def test_union_list_str_none(self, adapter_class):
        """Union[List[str], None] should be detected."""
        assert adapter_class.is_list_str(Union[List[str], None]) is True

    def test_plain_str_not_detected(self, adapter_class):
        """str should not be detected as list[str]."""
        assert adapter_class.is_list_str(str) is False

    def test_list_int_not_detected(self, adapter_class):
        """list[int] should not be detected as list[str]."""
        assert adapter_class.is_list_str(list[int]) is False

    def test_optional_str_not_detected(self, adapter_class):
        """Optional[str] should not be detected as list[str]."""
        assert adapter_class.is_list_str(Optional[str]) is False

    def test_optional_list_int_not_detected(self, adapter_class):
        """Optional[List[int]] should not be detected as list[str]."""
        assert adapter_class.is_list_str(Optional[List[int]]) is False

    def test_union_multiple_types_not_detected(self, adapter_class):
        """Union[List[str], int] should not be detected (multiple non-None types)."""
        assert adapter_class.is_list_str(Union[List[str], int]) is False


class TestTryParseListStr:
    """Tests for TolerantChatAdapter.try_parse_list_str() parsing."""

    @pytest.fixture
    def adapter_class(self):
        from minima_llm.dspy_adapter import TolerantChatAdapter
        return TolerantChatAdapter

    # === JSON (double-quote) tests ===

    def test_valid_json_array(self, adapter_class):
        """Valid JSON array should parse correctly."""
        result = adapter_class.try_parse_list_str('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_array_with_whitespace(self, adapter_class):
        """JSON array items should be stripped."""
        result = adapter_class.try_parse_list_str('["  a  ", "b", "c  "]')
        assert result == ["a", "b", "c"]

    def test_json_array_filters_empty(self, adapter_class):
        """Empty strings should be filtered out."""
        result = adapter_class.try_parse_list_str('["a", "", "b", null, "c"]')
        assert result == ["a", "b", "c"]

    def test_json_with_apostrophe(self, adapter_class):
        """JSON handles apostrophes in strings correctly."""
        result = adapter_class.try_parse_list_str('["What is the article\'s main point?"]')
        assert result == ["What is the article's main point?"]

    def test_invalid_json_raises(self, adapter_class):
        """Invalid input should raise ValueError."""
        with pytest.raises(ValueError, match="Expected JSON array"):
            adapter_class.try_parse_list_str("not json")

    def test_json_object_raises(self, adapter_class):
        """JSON object (not array) should raise ValueError."""
        with pytest.raises(ValueError, match="Expected JSON array"):
            adapter_class.try_parse_list_str('{"key": "value"}')

    def test_json_string_raises(self, adapter_class):
        """JSON string (not array) should raise ValueError."""
        with pytest.raises(ValueError, match="Expected JSON array"):
            adapter_class.try_parse_list_str('"just a string"')

    # === Python (single-quote) tests ===

    def test_python_single_quote_array(self, adapter_class):
        """Python-style single-quote list should parse correctly."""
        result = adapter_class.try_parse_list_str("['a', 'b', 'c']")
        assert result == ["a", "b", "c"]

    def test_python_single_quote_with_whitespace(self, adapter_class):
        """Python list items should be stripped."""
        result = adapter_class.try_parse_list_str("['  a  ', 'b', 'c  ']")
        assert result == ["a", "b", "c"]

    def test_python_with_escaped_apostrophe(self, adapter_class):
        """Python list with properly escaped apostrophe should parse."""
        result = adapter_class.try_parse_list_str("['What is the article\\'s main point?']")
        assert result == ["What is the article's main point?"]

    def test_unbalanced_quotes_raises(self, adapter_class):
        """Odd number of single quotes should raise ValueError (apostrophe issue)."""
        with pytest.raises(ValueError, match="Unbalanced single quotes"):
            adapter_class.try_parse_list_str("['What about article's thing']")

    # === Empty/edge cases ===

    def test_empty_list(self, adapter_class):
        """Empty list should parse correctly."""
        result = adapter_class.try_parse_list_str("[]")
        assert result == []

    def test_empty_list_with_spaces(self, adapter_class):
        """Empty list with whitespace should parse correctly."""
        result = adapter_class.try_parse_list_str("  [  ]  ")
        assert result == []


class TestTolerantAdapterParse:
    """Test TolerantChatAdapter.parse() handles missing fields and list types."""

    @pytest.fixture
    def adapter(self):
        """Create TolerantChatAdapter instance."""
        from minima_llm.dspy_adapter import TolerantChatAdapter
        return TolerantChatAdapter()

    @pytest.fixture
    def signature_with_optional_confidence(self):
        """Create a signature with optional confidence field."""
        import dspy

        class TestSignature(dspy.Signature):
            __doc__ = "Test signature with optional confidence."
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()
            confidence: Optional[float] = dspy.OutputField()

        return TestSignature

    @pytest.fixture
    def signature_with_required_fields(self):
        """Create a signature with all required fields."""
        import dspy

        class TestSignature(dspy.Signature):
            __doc__ = "Test signature with required fields."
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()
            reasoning: str = dspy.OutputField()

        return TestSignature

    @pytest.fixture
    def signature_with_list_str(self):
        """Create a signature with list[str] output."""
        import dspy

        class TestSignature(dspy.Signature):
            __doc__ = "Test signature with list output."
            query: str = dspy.InputField()
            questions: list[str] = dspy.OutputField()

        return TestSignature

    @pytest.fixture
    def signature_with_optional_list_str(self):
        """Create a signature with Optional[List[str]] output."""
        import dspy

        class TestSignature(dspy.Signature):
            __doc__ = "Test signature with optional list output."
            query: str = dspy.InputField()
            questions: Optional[List[str]] = dspy.OutputField()

        return TestSignature

    def test_missing_optional_field_returns_none(self, adapter, signature_with_optional_confidence):
        """Optional fields missing from output should get None."""
        completion = dedent("""
            [[ ## answer ## ]]
            The capital of France is Paris.
        """)
        result = adapter.parse(signature_with_optional_confidence, completion)
        assert result["answer"] == "The capital of France is Paris."
        assert result["confidence"] is None

    def test_present_optional_field_parsed(self, adapter, signature_with_optional_confidence):
        """Optional fields present should be parsed."""
        completion = dedent("""
            [[ ## answer ## ]]
            The capital of France is Paris.
            [[ ## confidence ## ]]
            0.95
        """)
        result = adapter.parse(signature_with_optional_confidence, completion)
        assert result["answer"] == "The capital of France is Paris."
        assert result["confidence"] == 0.95

    def test_missing_required_field_raises(self, adapter, signature_with_required_fields):
        """Required fields missing should raise AdapterParseError."""
        from minima_llm.dspy_adapter import AdapterParseError

        completion = dedent("""
            [[ ## answer ## ]]
            The capital of France is Paris.
        """)
        with pytest.raises(AdapterParseError, match="Missing required field: reasoning"):
            adapter.parse(signature_with_required_fields, completion)

    def test_list_str_field_parsed_from_json(self, adapter, signature_with_list_str):
        """list[str] output should be parsed from JSON."""
        completion = dedent("""
            [[ ## questions ## ]]
            ["What is the capital?", "What is the population?"]
        """)
        result = adapter.parse(signature_with_list_str, completion)
        assert result["questions"] == ["What is the capital?", "What is the population?"]

    def test_list_str_field_parsed_from_python(self, adapter, signature_with_list_str):
        """list[str] output should be parsed from Python syntax."""
        completion = dedent("""
            [[ ## questions ## ]]
            ['What is the capital?', 'What is the population?']
        """)
        result = adapter.parse(signature_with_list_str, completion)
        assert result["questions"] == ["What is the capital?", "What is the population?"]

    def test_list_str_with_apostrophe_in_json(self, adapter, signature_with_list_str):
        """list[str] containing apostrophes should parse from JSON."""
        completion = dedent("""
            [[ ## questions ## ]]
            ["What's the article's main point?", "How does it compare?"]
        """)
        result = adapter.parse(signature_with_list_str, completion)
        assert result["questions"] == ["What's the article's main point?", "How does it compare?"]

    def test_list_str_with_apostrophe_raises_in_python_syntax(self, adapter, signature_with_list_str):
        """list[str] with unescaped apostrophe in Python syntax should raise."""
        from minima_llm.dspy_adapter import AdapterParseError

        completion = dedent("""
            [[ ## questions ## ]]
            ['What's the main point?']
        """)
        with pytest.raises(AdapterParseError, match="Unbalanced single quotes"):
            adapter.parse(signature_with_list_str, completion)

    def test_optional_list_str_missing_returns_none(self, adapter, signature_with_optional_list_str):
        """Optional[List[str]] missing should return None."""
        completion = dedent("""
            [[ ## other_field ## ]]
            Some value
        """)
        result = adapter.parse(signature_with_optional_list_str, completion)
        assert result["questions"] is None

    def test_optional_list_str_present_parsed(self, adapter, signature_with_optional_list_str):
        """Optional[List[str]] present should be parsed."""
        completion = dedent("""
            [[ ## questions ## ]]
            ["Question 1", "Question 2"]
        """)
        result = adapter.parse(signature_with_optional_list_str, completion)
        assert result["questions"] == ["Question 1", "Question 2"]


class TestStockVsTolerantAdapter:
    """Compare DSPy 3.1 stock ChatAdapter vs TolerantChatAdapter."""

    @pytest.fixture
    def tolerant_adapter(self):
        from minima_llm.dspy_adapter import TolerantChatAdapter
        return TolerantChatAdapter()

    @pytest.fixture
    def stock_adapter(self):
        from dspy.adapters.chat_adapter import ChatAdapter
        return ChatAdapter()

    @pytest.fixture
    def simple_signature(self):
        """Simple signature with string output."""
        import dspy

        class TestSignature(dspy.Signature):
            __doc__ = "Simple test signature."
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        return TestSignature

    @pytest.fixture
    def signature_with_literal(self):
        """Signature with Literal output (like better_passage)."""
        import dspy

        class TestSignature(dspy.Signature):
            __doc__ = "Test signature with literal choice."
            passage_1: str = dspy.InputField()
            passage_2: str = dspy.InputField()
            better_passage: Literal["1", "2"] = dspy.OutputField()

        return TestSignature

    def test_both_adapters_parse_simple_output(self, tolerant_adapter, stock_adapter, simple_signature):
        """Both adapters should parse well-formed simple output identically."""
        completion = dedent("""
            [[ ## answer ## ]]
            The capital of France is Paris.
        """)

        tolerant_result = tolerant_adapter.parse(simple_signature, completion)
        stock_result = stock_adapter.parse(simple_signature, completion)

        assert tolerant_result["answer"] == "The capital of France is Paris."
        assert stock_result["answer"] == "The capital of France is Paris."

    def test_both_adapters_parse_literal_output(self, tolerant_adapter, stock_adapter, signature_with_literal):
        """Both adapters should parse Literal output."""
        completion = dedent("""
            [[ ## better_passage ## ]]
            1
        """)

        tolerant_result = tolerant_adapter.parse(signature_with_literal, completion)
        stock_result = stock_adapter.parse(signature_with_literal, completion)

        assert tolerant_result["better_passage"] == "1"
        assert stock_result["better_passage"] == "1"

    def test_stock_adapter_list_str_behavior(self, stock_adapter):
        """Document how stock ChatAdapter handles list[str] fields."""
        import dspy

        class ListSignature(dspy.Signature):
            __doc__ = "Signature with list output."
            query: str = dspy.InputField()
            items: list[str] = dspy.OutputField()

        completion = dedent("""
            [[ ## items ## ]]
            ["item1", "item2", "item3"]
        """)

        result = stock_adapter.parse(ListSignature, completion)

        if isinstance(result["items"], str):
            assert '["item1"' in result["items"]
        else:
            assert result["items"] == ["item1", "item2", "item3"]


class TestAdapterVersionSelection:
    """Test that the correct adapter is selected based on DSPy version."""

    def test_get_dspy_version_returns_tuple(self):
        """_get_dspy_version should return a 3-tuple of ints."""
        from minima_llm.dspy_adapter import _get_dspy_version

        version = _get_dspy_version()
        assert isinstance(version, tuple)
        assert len(version) == 3
        assert all(isinstance(v, int) for v in version)

    def test_select_adapter_returns_adapter(self):
        """_select_adapter should return an adapter instance."""
        from minima_llm.dspy_adapter import _select_adapter
        from dspy.adapters.chat_adapter import ChatAdapter

        adapter = _select_adapter()
        assert isinstance(adapter, ChatAdapter)

    def test_tolerant_adapter_is_chat_adapter_subclass(self):
        """TolerantChatAdapter should be a ChatAdapter subclass."""
        from minima_llm.dspy_adapter import TolerantChatAdapter
        from dspy.adapters.chat_adapter import ChatAdapter

        assert issubclass(TolerantChatAdapter, ChatAdapter)
