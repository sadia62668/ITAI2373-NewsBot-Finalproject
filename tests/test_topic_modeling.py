from src.analysis.topic_modeler import TopicModeler, topic_purity

DOCS = (["football match goal team player cup league"] * 10 + ["bank market share profit economy growth"] * 10
        + ["film award actor oscar star music"] * 10)


def test_nmf_and_lda_shapes():
    for method in ["nmf", "lda"]:
        tm = TopicModeler(n_topics=3, method=method).fit(DOCS)
        assert tm.doc_topics.shape == (30, 3)
        assert len(tm.top_words(5)) == 3


def test_topic_purity_range():
    tm = TopicModeler(n_topics=3, method="nmf").fit(DOCS)
    p = topic_purity(tm.topic_vs_category(["s"] * 10 + ["b"] * 10 + ["e"] * 10))
    assert 0 < p <= 1
