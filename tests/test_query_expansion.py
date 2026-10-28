from src.retrieval.query_expansion import FinancialAcronymExpander


def test_acronym_expander_single():
    expander = FinancialAcronymExpander()
    query = "How do I register a DBA for my business?"
    expanded, acronyms = expander.expand(query)

    assert "DBA (Doing Business As)" in expanded
    assert "DBA" in acronyms


def test_acronym_expander_multiple():
    expander = FinancialAcronymExpander()
    query = "Can my LLC use an EIN or should I get a HELOC?"
    expanded, acronyms = expander.expand(query)

    assert "LLC (Limited Liability Company)" in expanded
    assert "EIN (Employer Identification Number)" in expanded
    assert "HELOC (Home Equity Line of Credit)" in expanded
    assert set(acronyms) == {"LLC", "EIN", "HELOC"}


def test_acronym_expander_noop():
    expander = FinancialAcronymExpander()
    query = "How do I deposit money into my checking account?"
    expanded, acronyms = expander.expand(query)

    assert expanded == query
    assert len(acronyms) == 0

