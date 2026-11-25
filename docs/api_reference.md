# API Reference

All modules live under `src/`. Inputs marked *processed* must first go through `preprocess_text`.

## `src.data_processing`
- `text_preprocessor.clean_text(text) -> str` · `preprocess_text(text) -> str` · `split_sentences(text) -> list[str]`
- `data_validator.load_bbc(path) -> DataFrame` · `validate(df) -> DataFrame` (report in `df.attrs["validation_report"]`)
- `feature_extractor.build_tfidf(texts, **params)` · `tfidf_dataframe(...)` · `get_top_tfidf_terms(tfidf_df, category, n_terms)` · `analyze_pos_patterns(text)` · `extract_syntactic_features(text, nlp)` · `combined_features(tfidf_matrix, sentiment_df, texts)`

## `src.analysis`
- `classifier.NewsClassifier`: `compare_models(Xtr, ytr, Xte, yte)`, `fit(X, y, tune=True, c_grid=...)`, `evaluate(X, y) -> (acc, report)`, `predict(texts)`, `predict_with_confidence(text) -> (cat, prob)`, `predict_top_k(text, k)`, `save(path)`, `load(path)`; `make_pipeline(model)`
- `sentiment_analyzer.SentimentAnalyzer`: `analyze(text)`, `analyze_df(df, text_col)`, `sentiment_evolution(sent_df, time_col)`
- `topic_modeler.TopicModeler(n_topics=5, method="nmf"|"lda")`: `fit`, `fit_transform`, `transform`, `top_words(n)`, `get_topic_words(id, n)`, `dominant_topic()`, `topic_vs_category(cats)`, `topic_evolution(periods)`, `cluster_articles(n)`, `visualize_topics(n_words, save_path)`; `topic_purity(crosstab)`
- `ner_extractor`: `get_nlp(model)`, `extract_entities(text) -> [(text, label)]`, `entity_table(df, text_col, limit)`, `entity_relationships(ent_df, min_count, top_n)`

## `src.language_models`
- `summarizer.summarize(text, n_sentences=3, method="textrank"|"tfidf")`, `lead_baseline(text, n)`
- `embeddings.SemanticSearch(use_embeddings=False)`: `fit(texts)`, `search(query, top_k) -> [(idx, score)]`, `similar_to(idx, top_k)`, `expand_query(query, n_terms)`
- `generator.enhance_article(text, summary, category, confidence, sentiment, entities, related)`, `generate_insights(df, sent_df, topic_words)`

## `src.multilingual`
- `language_detector.detect_language(text) -> (code, prob)`
- `translator.translate(text, target="en", source="auto") -> (text, ok)`
- `cross_lingual_analyzer.MultilingualAnalyzer(classifier, sentiment)`: `analyze(text)`, `compare(texts)`, `coverage_comparison(stories)`; `RawTextClassifier(news_clf)`

## `src.conversation`
- `intent_classifier.detect_intent(query) -> (intent, slots)`, `split_multi_part(query)`, `find_category`, `find_sentiment_filter`
- `response_generator.ResponseGenerator(bot).respond(intent, slots, ctx)`
- `query_processor.QueryProcessor(bot).process(query)`

## `src.utils`
- `visualization`: `plot_category_distribution`, `plot_confusion_matrix`, `plot_pos_heatmap`, `plot_topic_evolution`, `plot_sentiment_evolution`
- `evaluation`: `search_precision_at_k`, `summary_label_preservation`, `intent_accuracy`
- `export.export_report(bot, path)`

## `src.newsbot.NewsBot2(df)`
`train(tune=True)`, `ask(query)`, `analyze_article(text)`, `summarize(text, n)`, `search(query, top_k)`, `topics(n)`, `insights()`, `export_report(path)`. Attributes after training: `classifier`, `test_accuracy`, `sentiment`, `sent_df`, `topic_model`, `searcher`, `multi`, `chat`.
